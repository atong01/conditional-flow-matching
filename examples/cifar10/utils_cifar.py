import os
import torch
from torchdyn.core import NeuralODE
from tqdm import tqdm

from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_samples(
    node_, model, savedir, step, net_="normal", batch_size=512, num_gen=1000
):
    model.eval()
    gen_path = savedir + f"{net_}_generated_FM_images_step_{step}.png"
    for batch in tqdm(range(num_gen // batch_size + 1)):
        with torch.no_grad():
            traj = node_.trajectory(
                torch.randn(64, 3, 32, 32).to(device),
                t_span=torch.linspace(0, 1, 100).to(device),
            )
            traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
            traj = traj / 2 + 0.5
        for i in range(batch_size):
            idx = batch * batch_size + i
            if idx < num_gen:
                img_path = os.path.join(gen_path, f"{idx}.png")
                save_image(traj[i], img_path)
    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x
