import os

import matplotlib.pyplot as plt
import torch
import torchsde
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
from timm import scheduler

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet.unet import UNetModelWrapper

from cleanfid import fid

savedir = "weights/"
os.makedirs(savedir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
batch_size = 10
n_epochs = 20

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)


####
# LOAD MODEL
###
new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=256,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0).to(device)

new_net = torch.nn.DataParallel(new_net).cuda() 

PATH = 'weights/reproduced/reproduced_cifar10_weights_epoch_999.pt'
print("path: ", PATH)
checkpoint = torch.load(PATH)
new_net.load_state_dict(checkpoint['model_state_dict'])

new_net = new_net.module.to(device)
new_net.eval()

####
# Define ODE
###
node = NeuralODE(new_net, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

def gen_1_img(unused_latent):
    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(32, 3, 32, 32).to(device),
            t_span=torch.linspace(0, 1, 100).to(device),
        )
    traj = traj[-1, :]#.view([-1, 3, 32, 32]).clip(-1, 1)
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)#.permute(1, 2, 0)
    return img

score = fid.compute_fid(gen=gen_1_img, dataset_name="cifar10", dataset_res=32, num_gen=50_000, dataset_split="train", mode='legacy_tensorflow')

print(score)
