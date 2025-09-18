import torch
from torchcfm.models import MLP
from torchcfm.models.unet import UNetModel


def test_initialize_models():
    UNetModel(
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        num_classes=10,
        class_cond=True,
    )
    model = MLP(dim=2, time_varying=True, w=64)


def test_conditional_model_without_integer_labels():

    model = UNetModel(
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        class_cond=False,
    )
    batch = torch.ones((8,28,28), dtype=torch.float32)
    label = 42.1*torch.ones((8,)).float()
    timesteps = torch.linspace(0,1,steps=5)

    #forward run
    x_ = model(t= timesteps, x=batch, y=label)
