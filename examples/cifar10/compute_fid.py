import os
import sys

import matplotlib.pyplot as plt
from absl import app, flags
import torch
from torchdyn.core import NeuralODE

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet.unet import UNetModelWrapper

from cleanfid import fid


FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_bool("parallel", False, help="multi gpu training")

FLAGS(sys.argv)

print('num_channel: ', FLAGS.num_channel)
#savedir = "results/cifar10_weights_step_400000.pt"
#os.makedirs(savedir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

print("loading model")
####
# LOAD MODEL
###
new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1).to(device)

print("model loaded")
#if FLAGS.parallel:
#    new_net = torch.nn.DataParallel(new_net).cuda() 


PATH = 'results/cifar10_weights_step_400000.pt'
print("path: ", PATH)
checkpoint = torch.load(PATH)
new_net.load_state_dict(checkpoint['ema_model'])

#if FLAGS.parallel:
#    new_net = new_net.module.to(device)

new_net.eval()

####
# Define ODE
###
node = NeuralODE(new_net, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

def gen_1_img(unused_latent):
    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(500, 3, 32, 32).to(device),
            t_span=torch.linspace(0, 1, 100).to(device),
        )
    traj = traj[-1, :]#.view([-1, 3, 32, 32]).clip(-1, 1)
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)#.permute(1, 2, 0)
    return img

print("Start computing FID")
score = fid.compute_fid(gen=gen_1_img, dataset_name="cifar10", batch_size=500, dataset_res=32, num_gen=50_000, dataset_split="train", mode='legacy_tensorflow')

print('FID: ', score)
