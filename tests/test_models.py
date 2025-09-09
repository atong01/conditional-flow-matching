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
    MLP(dim=2, time_varying=True, w=64)
