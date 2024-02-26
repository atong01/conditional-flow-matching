import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper
from sklearn.preprocessing import StandardScaler

from utils_single_cell import adata_dataset, split, combined_loader, train_dataloader, val_dataloader, test_dataloader

adata = sc.read_h5ad("./ebdata_v2.h5ad")
max_dim=1000

data, labels, ulabels = adata_dataset("./ebdata_v2.h5ad")

if max_dim==1000:
    sc.pp.highly_variable_genes(adata, n_top_genes=max_dim)
    adata = adata.X[:, adata.var["highly_variable"]].toarray()

# Standardize coordinates
print(adata.shape)
scaler = StandardScaler()
scaler.fit(adata)
data = scaler.transform(adata)

dim = data.shape[-1]
print("data dim: ", dim)

timepoint_data = [
    adata[labels == lab].astype(np.float32) for lab in ulabels
]
    
split_timepoint_data = split(timepoint_data)

print(f"Loaded ebdata with timepoints {ulabels} of sizes {[len(d) for d in timepoint_data]} with dim {dim}.")


#### TRAINING
train_dataloader = train_dataloader(split_timepoint_data)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 128
sigma = 0.1
ot_cfm_model = MLP(dim=dim, time_varying=True, w=256).to(device)
ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), 1e-4)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

n_epochs = 2000
for _ in range(n_epochs):
    for X in train_dataloader:
        t_snapshot = np.random.randint(0,4)
        t, xt, ut = FM.sample_location_and_conditional_flow(X[t_snapshot], X[t_snapshot+1])
        ot_cfm_optimizer.zero_grad()
        vt = ot_cfm_model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        ot_cfm_optimizer.step()