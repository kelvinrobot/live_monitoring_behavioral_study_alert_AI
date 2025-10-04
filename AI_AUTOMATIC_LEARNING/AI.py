
import random, uuid
from datetime import datetime, timedelta, date
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import networkx as nx


import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib, os

DEVICE = 'cpu'
SAVE_DIR = "/content/drive/MyDrive/all_five_models/trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)


# 1. Transformer Training
print("\n[1] Training Transformer...")
transformer = transformer.to(DEVICE)
opt = torch.optim.Adam(transformer.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for ep in range(5):
    transformer.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out, _ = transformer(xb)
        loss = loss_fn(out, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {ep+1}, loss {loss.item():.4f}")

# Eval
transformer.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        out, _ = transformer(xb.to(DEVICE))
        probs = F.softmax(out, dim=1)[:,1].cpu().numpy()
        all_preds.extend(probs); all_labels.extend(yb.numpy())
print("Transformer ROC-AUC:", roc_auc_score(all_labels, all_preds))
print("Transformer PR-AUC:", average_precision_score(all_labels, all_preds))
torch.save(transformer.state_dict(), f"{SAVE_DIR}/transformer.pt")

# 2. VAE (unsupervised)

print("\n[2] Training VAE...")
vae = vae.to(DEVICE)
vae_opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

def vae_loss_fn(recon, x, mu, logvar):
    BCE = F.mse_loss(recon, x, reduction="mean")
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

for ep in range(5):
    vae.train()
    for xb, _ in train_loader:
        xb = xb.to(DEVICE)
        recon, mu, lv = vae(xb)
        loss = vae_loss_fn(recon, xb, mu, lv)
        vae_opt.zero_grad(); loss.backward(); vae_opt.step()
    print(f"Epoch {ep+1}, loss {loss.item():.4f}")

torch.save(vae.state_dict(), f"{SAVE_DIR}/vae.pt")

# 3. Contrastive Encoder

print("\n[3] Training Contrastive Encoder...")
contrastive = contrastive.to(DEVICE)
contrastive_opt = torch.optim.Adam(contrastive.parameters(), lr=1e-3)

def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim = sim / temperature
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels, labels], dim=0)
    return F.cross_entropy(sim, labels)

for ep in range(3):
    contrastive.train()
    for xb,_ in train_loader:
        xb = xb.to(DEVICE)
        z1, _ = contrastive(xb)
        noise = xb + 0.01*torch.randn_like(xb)
        z2, _ = contrastive(noise)
        loss = contrastive_loss(z1, z2)
        contrastive_opt.zero_grad(); loss.backward(); contrastive_opt.step()
    print(f"Epoch {ep+1}, loss {loss.item():.4f}")

torch.save(contrastive.state_dict(), f"{SAVE_DIR}/contrastive.pt")


# 4. Graph GCN

print("\n[4] Training GCN...")
X_nodes, A_norm, node_list, emp_nodes, res_nodes, y_emp = prepare_gcn(G_recent, agg_df)
X_nodes, A_norm, y_emp = X_nodes.to(DEVICE), A_norm.to(DEVICE), y_emp.to(DEVICE)

gcn = gcn.to(DEVICE)
gcn_opt = torch.optim.Adam(gcn.parameters(), lr=1e-3)
clf_layer = nn.Linear(24, 2).to(DEVICE)

for ep in range(5):
    gcn.train()
    emb = gcn(X_nodes, A_norm)
    logits = clf_layer(emb[:len(emp_nodes)])
    loss = F.cross_entropy(logits, y_emp)
    gcn_opt.zero_grad(); loss.backward(); gcn_opt.step()
    print(f"Epoch {ep+1}, loss {loss.item():.4f}")

torch.save({
    "gcn": gcn.state_dict(),
    "clf": clf_layer.state_dict(),
    "node_list": node_list,
    "emp_nodes": emp_nodes
}, f"{SAVE_DIR}/gcn.pt")


# 5. Tabular MLP

print("\n[5] Training Tabular MLP...")
tab_mlp = tab_mlp.to(DEVICE)
tab_opt = torch.optim.Adam(tab_mlp.parameters(), lr=1e-3)
clf_tab = nn.Linear(24,2).to(DEVICE)

Xtr = torch.tensor(X_tab_train, dtype=torch.float32).to(DEVICE)
ytr = torch.tensor(y_tab_train, dtype=torch.long).to(DEVICE)
Xva = torch.tensor(X_tab_val, dtype=torch.float32).to(DEVICE)
yva = torch.tensor(y_tab_val, dtype=torch.long).to(DEVICE)

for ep in range(5):
    tab_mlp.train()
    emb = tab_mlp(Xtr)
    logits = clf_tab(emb)
    loss = F.cross_entropy(logits, ytr)
    tab_opt.zero_grad(); loss.backward(); tab_opt.step()
    print(f"Epoch {ep+1}, loss {loss.item():.4f}")

tab_mlp.eval()
with torch.no_grad():
    emb = tab_mlp(Xva); logits = clf_tab(emb)
    probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
print("Tabular ROC-AUC:", roc_auc_score(yva.cpu(), probs))
print("Tabular PR-AUC:", average_precision_score(yva.cpu(), probs))

torch.save({
    "mlp": tab_mlp.state_dict(),
    "clf": clf_tab.state_dict()
}, f"{SAVE_DIR}/tabular.pt")

# Save preprocessor
joblib.dump(encoders, f"{SAVE_DIR}/encoders.pkl")
joblib.dump(scaler_tab, f"{SAVE_DIR}/scaler.pkl")

print("\n All models trained & saved in:", SAVE_DIR)
