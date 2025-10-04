# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import os


# Paths

BASE_DIR = Path(__file__).resolve().parent   
TRANSFORMER_DIR = BASE_DIR /"all_five_models" /"trained_models"/"transformer.pt"
CONTRASTIVE_DIR = BASE_DIR /"all_five_models" /"trained_models"/"contrastive.pt"
GCN_DIR = BASE_DIR /"all_five_models" /"trained_models"/"gcn.pt"
VAE_DIR = BASE_DIR /"all_five_models" /"trained_models"/"vae.pt"
TABULAR_DIR = BASE_DIR /"all_five_models" /"trained_models"/"tabular.pt"
DEVICE = "cpu"

# Model Definitions


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, nhead=2, num_layers=2, num_classes=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        h = self.transformer(x)
        out = self.fc(h.mean(1))
        return out

class SequenceVAE(nn.Module):
    def __init__(self, input_dim=6, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc21 = nn.Linear(32, latent_dim)
        self.fc22 = nn.Linear(32, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, latent_dim)

    def forward(self, x):
        return F.normalize(self.fc2(F.relu(self.fc1(x))), dim=-1)

class SimpleGCN(nn.Module):
    def __init__(self, in_dim=6, hid=16, out_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, X, A):
        h = torch.matmul(A, X)
        h = F.relu(self.fc1(h))
        h = torch.matmul(A, h)
        return self.fc2(h)

class TabularMLP(nn.Module):
    def __init__(self, input_dim=6, hidden=32, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# Safe model loading helper

def safe_load(model, path):
    if not path.exists():
        #st.warning(f" Missing model: {path.name}")
        return model
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


# Load Models

transformer = safe_load(TransformerClassifier(), TRANSFORMER_DIR)
vae = safe_load(SequenceVAE(), VAE_DIR)
contrastive = safe_load(ContrastiveEncoder(), CONTRASTIVE_DIR)
gcn = safe_load(SimpleGCN(), GCN_DIR)
mlp = safe_load(TabularMLP(), TABULAR_DIR)

transformer.eval(); vae.eval(); contrastive.eval(); gcn.eval(); mlp.eval()


# Streamlit Dashboard

st.set_page_config(page_title="Bank Employee Monitoring AI", layout="wide")
st.title("🏦 Bank Employee Monitoring AI Dashboard")

st.sidebar.header("Simulation Controls")
num_employees = st.sidebar.slider("Number of employees", 5, 50, 10)
simulate_btn = st.sidebar.button("Run Simulation")

# Synthetic Data Generator

def generate_employee_activity(n=10, seq_len=12):
    data = []
    for emp_id in range(1, n+1):
        accesses = np.random.randint(0, 50, seq_len)
        tx_amounts = np.random.exponential(1000, seq_len)
        durations = np.random.normal(5, 2, seq_len)
        critical_access = np.random.binomial(1, 0.1, seq_len)
        hour = np.random.randint(0, 24, seq_len)
        loc = np.random.randint(0, 5, seq_len)
        seq = np.vstack([accesses, tx_amounts, durations, critical_access, hour, loc]).T
        data.append((emp_id, seq))
    return data

# Run Simulation

if simulate_btn:
    st.subheader(" Employee Activity Simulation")
    employees = generate_employee_activity(num_employees)

    results = []
    for emp_id, seq in employees:
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        # Transformer prediction
        pred = torch.softmax(transformer(x), dim=-1)[0,1].item()

        # VAE reconstruction error
        recon, _, _ = vae(x)
        vae_loss = F.mse_loss(recon, x).item()

        # Contrastive embedding anomaly score
        emb = contrastive(x).mean(0)
        contrastive_score = torch.norm(emb).item()

        # Tabular summary
        daily = torch.tensor(seq.mean(axis=0), dtype=torch.float32)
        tab_pred = torch.softmax(mlp(daily), dim=-1)[1].item()

        results.append({
            "Employee": emp_id,
            "Transformer Risk": pred,
            "VAE Recon Error": vae_loss,
            "Contrastive Score": contrastive_score,
            "Tabular Risk": tab_pred
        })

    df = pd.DataFrame(results)
    st.dataframe(df.style.highlight_max(axis=0, color="red"))

    # Risk distribution
    st.subheader(" Risk Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Transformer Risk"], alpha=0.6, label="Transformer")
    ax.hist(df["Tabular Risk"], alpha=0.6, label="Tabular MLP")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    # Graph visualization
    st.subheader(" Graph Neural Net Employee-Resource View")
    G = nx.erdos_renyi_graph(num_employees, 0.2, seed=42)
    nx.draw(G, with_labels=True, node_color="skyblue", node_size=800, font_size=10)
    st.pyplot(plt.gcf())
