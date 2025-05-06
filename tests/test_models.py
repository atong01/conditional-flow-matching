import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models import MLP
from torchcfm.models.unet import UNetModel


def test_initialize_unet():
    model = UNetModel(
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        num_classes=10,
        class_cond=True,
    )

    batch = torch.zeros((8, 1, 28, 28), dtype=torch.float32)
    label = torch.ones((8,), dtype=torch.long)
    timesteps = torch.linspace(0,1,steps=8)

    _ = model(t= timesteps, x=batch, y=label)


def test_initialize_mlp():

    model1 = MLP(dim=2, time_varying=True, w=64)
    batch = torch.ones((8, 3), dtype=torch.float32)
    output1 = model1(x=batch)

    assert output1.shape == (8,2)

    model2 = MLP(dim=2, w=64)
    batch = torch.ones((8, 2), dtype=torch.float32)
    output2 = model2(x=batch)

    assert output2.shape == (8,2)


class mock_embedding(torch.nn.Module):

    def __init__(self, outdim=128):
        super().__init__()
        self.outdim = outdim

    def forward(self, inputs):

        batchsize = inputs.size(0)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape((batchsize, 1))

        return torch.tile(inputs, (1, self.outdim))


def test_conditional_model_without_integer_labels():

    model_channels = 32
    model = UNetModel(
        dim=(1, 28, 28),
        num_channels=model_channels,
        num_res_blocks=1,
        class_cond=False,
        embedding_net = mock_embedding
    )

    x1 = torch.ones((8, 1, 28, 28), dtype=torch.float32)
    x0 = torch.randn_like(x1)
    FM = ConditionalFlowMatcher(sigma=0.)
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    label = 42.1*torch.ones((8,)).float()

    vt = model(t=t, x=xt, y=label)
>>>>>>> 1dbfbd0 (more tests and floating point conditions)
