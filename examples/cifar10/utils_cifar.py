import torch
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_samples(model, parallel, savedir, step, net_="normal"):
    model.eval()
    if parallel:
        model = model.module.to(device)
    node_ = NeuralODE(model, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32).to(device),
            t_span=torch.linspace(0, 1, 100).to(device),
        )
    traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
    traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

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

            
class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, input_size=(3, 32, 32), reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.reverse = reverse

    # Drift
    def f(self, t, y):
        y = y.view(-1, 3, 32, 32)
        if self.reverse:
            t = 1 - t
            return -self.drift(t, y) + self.score(t, y)
        return self.drift(t, y).flatten(start_dim=1) - self.score(t, y).flatten(start_dim=1)

    # Diffusion
    def g(self, t, y):
        y = y.view(-1, 3, 32, 32)
        return (torch.ones_like(t) * torch.ones_like(y)).flatten(start_dim=1) * sigma