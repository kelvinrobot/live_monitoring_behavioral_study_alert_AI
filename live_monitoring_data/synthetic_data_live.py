
import random, uuid
from datetime import datetime, timedelta, date
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import networkx as nx

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")
print("Device:", DEVICE)

#  Small synthetic generator
ROLES = ['teller','loan_officer','back_office','admin','manager']
APPS = ['core_banking','payments','crm','reports','admin_console']
RESOURCES = [f'resource_{i:03d}' for i in range(80)]
ACTIONS_BY_ROLE = {
    'teller': ['login','view_account','deposit','withdraw','export_csv','logout'],
    'loan_officer': ['login','view_loan','edit_loan','approve_loan','export_csv','logout'],
    'back_office': ['login','run_report','export_csv','logout'],
    'admin': ['login','change_role','access_admin','logout','export_csv'],
    'manager': ['login','view_reports','approve_txn','logout']
}

def gen_pool(n=60):
    out=[]
    for i in range(n):
        out.append({'employee_id':f'emp_{i:03d}','role':random.choice(ROLES),'home_branch':f'br_{random.randint(1,20):02d}'})
    return out
EMP_POOL = gen_pool(60)
EMP_IDS = [e['employee_id'] for e in EMP_POOL]

def sample_session(emp, date):
    role = emp['role']
    n_events = np.random.randint(3,8)
    session_id = str(uuid.uuid4())
    start_hour = int(np.clip(np.random.normal(10 if role!='admin' else 14, 2), 6, 22))
    t = datetime.combine(date, datetime.min.time()) + timedelta(hours=start_hour, minutes=random.randint(0,59))
    events=[]
    for _ in range(n_events):
        action = random.choice(ACTIONS_BY_ROLE[role])
        app = random.choice(APPS)
        resource = random.choice(RESOURCES)
        duration_ms = int(np.clip(np.random.exponential(150), 20, 3000))
        bytes_out = int(np.random.exponential(1200)) if action=='export_csv' else 0
        tx_amount = float(np.round(np.random.lognormal(mean=8, sigma=1.0),2)) if action in ('deposit','withdraw','approve_txn','edit_loan') else 0.0
        events.append({'timestamp':t.isoformat(),'employee_id':emp['employee_id'],'role':role,'branch':emp['home_branch'],
                       'device_id':f'dev_{random.randint(1,300):03d}','app':app,'action':action,'resource':resource,
                       'result':'success' if random.random()>0.03 else 'failure','duration_ms':duration_ms,
                       'bytes_out':bytes_out,'tx_amount':tx_amount,'session_id':session_id})
        t += timedelta(seconds=random.randint(20,400))
    return events

def inject_attacks(df, n_attackers=4):
    attackers = random.sample(EMP_IDS, n_attackers)
    for a in attackers:
        mask = df['employee_id']==a
        if mask.sum()==0: continue
        idxs = df[mask].sample(n=min(4, mask.sum()), replace=True).index
        df.loc[idxs,'action'] = 'export_csv'
        df.loc[idxs,'bytes_out'] = df.loc[idxs,'bytes_out'] + np.random.randint(20000,80000, size=len(idxs))
        df.loc[idxs,'label'] = 1
    df['label'] = df['label'].fillna(0).astype(int)
    return df

def gen_hist(days=10):
    all_events=[]
    start=date(2024,9,1)
    for d_offset in range(days):
        d=start+timedelta(days=d_offset)
        for emp in EMP_POOL:
            if random.random()<0.12: continue
            n_sessions = np.random.poisson(1.0)+1
            for _ in range(n_sessions):
                all_events.extend(sample_session(emp,d))
    df = pd.DataFrame(all_events)
    df = inject_attacks(df, n_attackers=4)
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df=df.sort_values('timestamp').reset_index(drop=True)
    return df

print("Generating data...")
df_hist = gen_hist(10)
print("Events:", len(df_hist))

# Encoding & sequence construction
cat_cols = ['employee_id','role','branch','device_id','app','action','resource']
encoders = {}
for c in cat_cols:
    le = LabelEncoder(); df_hist[c] = df_hist[c].astype(str)
    le.fit(df_hist[c])
    encoders[c]=le
    df_hist[c+'_enc']=le.transform(df_hist[c])

SEQ_MAX_LEN=12
def build_session_sequences(df):
    sessions=[]; labels=[]; meta=[]
    for sid,g in df.groupby('session_id'):
        g=g.sort_values('timestamp')
        vecs=[]
        for _,row in g.iterrows():
            vec=[int(row['action_enc']), int(row['app_enc']), int(row['resource_enc']), float(row['duration_ms']), float(row['bytes_out']), float(row['tx_amount'])]
            vecs.append(vec)
        if len(vecs)==0: continue
        if len(vecs)<SEQ_MAX_LEN:
            pad=[[0,0,0,0.0,0.0,0.0]]*(SEQ_MAX_LEN-len(vecs))
            vecs=pad+vecs
        else:
            vecs=vecs[-SEQ_MAX_LEN:]
        sessions.append(np.array(vecs,dtype=np.float32))
        labels.append(int(g['label'].max()))
        meta.append({'session_id':sid,'employee_id':g['employee_id'].iloc[0],'start_time':g['timestamp'].iloc[0]})
    return np.stack(sessions), np.array(labels,dtype=np.int64), pd.DataFrame(meta)

X_seq, y_seq, meta_sessions = build_session_sequences(df_hist)
print("Seq shape:", X_seq.shape, "positives:", y_seq.sum())

# Aggregates & graph snapshot
TAB_FEATURES=['events_count','unique_resources','exports_sum','avg_duration','total_tx']
def build_aggregates(df):
    df['date']=df['timestamp'].dt.date
    agg=df.groupby(['employee_id','date']).agg(events_count=('session_id','count'),
                                              unique_resources=('resource','nunique'),
                                              exports_sum=('bytes_out','sum'),
                                              avg_duration=('duration_ms','mean'),
                                              total_tx=('tx_amount','sum'),
                                              label=('label','max')).reset_index().fillna(0)
    agg['employee_enc']=encoders['employee_id'].transform(agg['employee_id'])
    return agg
agg_df = build_aggregates(df_hist)

def build_graph_snapshot(df, days=7):
    cutoff = df['timestamp'].max() - pd.Timedelta(days=days)
    rec = df[df['timestamp']>=cutoff]
    G=nx.Graph()
    for emp in rec['employee_id'].unique(): G.add_node(f"E_{emp}", bipartite=0)
    for res in rec['resource'].unique(): G.add_node(f"R_{res}", bipartite=1)
    edge_weights = rec.groupby(['employee_id','resource']).size().reset_index(name='count')
    for _,r in edge_weights.iterrows():
        G.add_edge(f"E_{r['employee_id']}", f"R_{r['resource']}", weight=int(r['count']))
    return G
G_recent = build_graph_snapshot(df_hist, days=7)

# Torch datasets
class SessionSeqDS(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X); self.y=torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=SEED, stratify=y_seq if y_seq.sum()>1 else None)
train_loader = DataLoader(SessionSeqDS(X_train,y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(SessionSeqDS(X_val,y_val), batch_size=64, shuffle=False)

X_tab = agg_df[TAB_FEATURES].values.astype(np.float32); y_tab = agg_df['label'].values.astype(np.int64)
X_tab_train, X_tab_val, y_tab_train, y_tab_val = train_test_split(X_tab,y_tab,test_size=0.2,random_state=SEED,stratify=y_tab if y_tab.sum()>1 else None)
scaler_tab = StandardScaler().fit(X_tab_train); X_tab_train = scaler_tab.transform(X_tab_train); X_tab_val = scaler_tab.transform(X_tab_val)

# Models
class SimpleTransformerClassifier(nn.Module):
    def __init__(self,input_dim,d_model=48,nhead=4,num_layers=1,num_classes=2):
        super().__init__()
        self.input_proj=nn.Linear(input_dim,d_model)
        enc=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer=nn.TransformerEncoder(enc, num_layers=num_layers)
        self.classifier=nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,num_classes))
    def forward(self,x):
        x=self.input_proj(x)
        x=self.transformer(x)
        x=x.mean(dim=1)
        return self.classifier(x), x

class LSTMVAE(nn.Module):
    def __init__(self,input_dim,hidden_dim=48,latent_dim=24):
        super().__init__()
        self.enc=nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu=nn.Linear(hidden_dim, latent_dim); self.fc_logvar=nn.Linear(hidden_dim, latent_dim)
        self.dec=nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.out=nn.Linear(hidden_dim, input_dim)
    def encode(self,x):
        _,(h,_) = self.enc(x); h=h[-1]
        return self.fc_mu(h), self.fc_logvar(h)
    def reparam(self,mu,logvar):
        std=torch.exp(0.5*logvar); eps=torch.randn_like(std); return mu+eps*std
    def decode(self,z,seq_len):
        z_seq = z.unsqueeze(1).repeat(1,seq_len,1); out,_=self.dec(z_seq); return self.out(out)
    def forward(self,x):
        mu,lv=self.encode(x); z=self.reparam(mu,lv); return self.decode(z,x.size(1)), mu, lv

class ContrastiveEncoder(nn.Module):
    def __init__(self,input_dim,emb_dim=48,proj_dim=24):
        super().__init__()
        self.input_proj=nn.Linear(input_dim,emb_dim)
        enc=nn.TransformerEncoderLayer(d_model=emb_dim,nhead=4, batch_first=True)
        self.transformer=nn.TransformerEncoder(enc,num_layers=1)
        self.proj=nn.Sequential(nn.Linear(emb_dim,emb_dim), nn.ReLU(), nn.Linear(emb_dim,proj_dim))
    def forward(self,x):
        x=self.input_proj(x)
        x=self.transformer(x)
        x=x.mean(dim=1)
        z=self.proj(x); return F.normalize(z,dim=1), x

class SimpleGCN(nn.Module):
    def __init__(self,in_dim, hid=48, out_dim=24):
        super().__init__(); self.fc1=nn.Linear(in_dim,hid); self.fc2=nn.Linear(hid,out_dim)
    def forward(self,X,A):
        h = torch.matmul(A, X); h=F.relu(self.fc1(h)); h=torch.matmul(A,h); return self.fc2(h)

class TabularMLP(nn.Module):
    def __init__(self,in_dim,out_dim=24): super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,64),nn.ReLU(),nn.Linear(64,out_dim))
    def forward(self,x): return self.net(x)

# instantiate
input_dim = X_seq.shape[2]
transformer = SimpleTransformerClassifier(input_dim=input_dim,d_model=48)
vae = LSTMVAE(input_dim,hidden_dim=48,latent_dim=24)
contrastive = ContrastiveEncoder(input_dim=input_dim,emb_dim=48,proj_dim=24)
gcn = SimpleGCN(in_dim=6,hid=48,out_dim=24)
tab_mlp = TabularMLP(in_dim=len(TAB_FEATURES), out_dim=24)
fusion_head = nn.Sequential(nn.Linear(48+24+24,64), nn.ReLU(), nn.Linear(64,2))

#  prepare GCN inputs with float32
def prepare_gcn(G, agg_df):
    emp_nodes = [n for n in G.nodes() if n.startswith("E_")]
    res_nodes = [n for n in G.nodes() if n.startswith("R_")]
    node_list = emp_nodes + res_nodes
    node_index = {n:i for i,n in enumerate(node_list)}
    n = len(node_list)
    node_feats = np.zeros((n,6), dtype=np.float32)
    emp_stats = agg_df.groupby('employee_id')[TAB_FEATURES].mean().to_dict(orient='index')
    for i,en in enumerate(emp_nodes):
        emp_id = en.replace("E_","")
        if emp_id in emp_stats:
            vals = np.array([emp_stats[emp_id].get(f,0.0) for f in TAB_FEATURES] + [0.0], dtype=np.float32)[:6]
            node_feats[i,:]=vals
    for j,rn in enumerate(res_nodes, start=len(emp_nodes)):
        idx = int(rn.split('_')[-1])
        rng = np.random.RandomState(idx); node_feats[j,:]=rng.normal(scale=1.0, size=6).astype(np.float32)
    A = nx.to_numpy_array(G, nodelist=node_list).astype(np.float32)
    A = A + np.eye(n, dtype=np.float32); D=A.sum(axis=1,keepdims=True); D[D==0]=1.0; A_norm = (A / D).astype(np.float32)
    emp_labels = [int(agg_df[agg_df['employee_id']==en.replace("E_","")]['label'].max() if en.replace("E_","") in agg_df['employee_id'].values else 0) for en in emp_nodes]
    return torch.tensor(node_feats, dtype=torch.float32), torch.tensor(A_norm, dtype=torch.float32), node_list, emp_nodes, res_nodes, torch.tensor(emp_labels,dtype=torch.long)

# training loops 
