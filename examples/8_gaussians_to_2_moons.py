import os
#################################
# TOY FOR SANITY CHECK
#################################

import math

import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/campus/kilian.fatras@MAIL.MCGILL.CA/flow_matching/conditional-flow-matching-private/')

from torchcfm.utils import *
from torchcfm.models import *
from torchcfm.conditional_flow_matching import *

savedir = "models/8gaussian-moons"
os.makedirs(savedir, exist_ok=True)
import torchsde

#################################
#            SF2M
#################################

sigma = 0.1
dim = 2
batch_size = 64
model = MLP(dim=dim, time_varying=True, w=64)
score_model = MLP(dim=dim, time_varying=True, w=64) 
optimizer = torch.optim.Adam(list(model.parameters()) + list(score_model.parameters()), 0.01)
FM = SF2M(sigma=0.1)

start = time.time()
max_k = 40000


for k in range(max_k):
    if (k+1)%1000==0:
        print(k)
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size)
    x1 = sample_moons(batch_size)
    x0, x1 = FM.entropic_ot_sampler.sample_plan(x0, x1)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t], dim=-1))
    FM_loss = torch.mean((vt - ut) ** 2)
    
    st = score_model(torch.cat([xt, t], dim=-1))
    score_target = FM.compute_score(x0, x1, t, xt)
    sigma_t = FM.compute_sigma_t(x0, x1, t)
    s_loss = torch.mean((sigma_t * st - score_target) ** 2)
    
    loss = s_loss + FM_loss
    
    loss.backward()
    optimizer.step()

    if (k + 1) % int(max_k/4) == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        with torch.no_grad():
            traj = node.trajectory(
                sample_8gaussians(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            plot_trajectories(traj)
            plt.title('SF2M ODE, batch_size={}'.format(batch_size))
            plt.savefig('results/sf2m_entropic_generated_samples_{}_iter_batch_size_{}'.format(k,batch_size))

        sde = SDE(model, score_model)
        with torch.no_grad():
            traj = torchsde.sdeint(
                sde,
                sample_8gaussians(1024),
                ts=torch.linspace(0, 1, 100),
                solver="euler",
            )
            plot_trajectories(traj)
            plt.title('SF2M SDE, batch_size={}'.format(batch_size))
            plt.savefig('results/sf2m_entropic_generated_samples_{}_iter_SDE_batch_size_{}'.format(k,batch_size))


torch.save(model, f"{savedir}/cfm_v1.pt")

#################################
# Conditional FM
#################################

sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=0.1)

start = time.time()
for k in range(20000):
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size)
    x1 = sample_moons(batch_size)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t], dim=-1))
    loss = torch.mean((vt - ut) ** 2)
    
    loss.backward()
    optimizer.step()

    if (k + 1) % 5000 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        with torch.no_grad():
            traj = node.trajectory(
                sample_8gaussians(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            plot_trajectories(traj)
            plt.savefig('results/cfm_generated_samples_{}_iter'.format(k))
torch.save(model, f"{savedir}/cfm_v1.pt")



#################################
#            OT-CFM
#################################

sigma = 0.1
dim = 2
batch_size = 256
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)

start = time.time()
for k in range(20000):
    optimizer.zero_grad()

    x0 = sample_8gaussians(batch_size)
    x1 = sample_moons(batch_size)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t], dim=-1))
    loss = torch.mean((vt - ut) ** 2)
    
    loss.backward()
    optimizer.step()

    if (k + 1) % 5000 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        with torch.no_grad():
            traj = node.trajectory(
                sample_8gaussians(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            plot_trajectories(traj)
            plt.savefig('results/ot_cfm_generated_samples_{}_iter'.format(k))
torch.save(model, f"{savedir}/cfm_v1.pt")


