import pytest
import torch
from torchcfm.diffusion_distance import HeatKernelKNN, torch_knn_from_data

def gt_heat_kernel_knn(
    data,
    t,
    k,
):
    L = torch_knn_from_data(data, k=k, projection=False, proj_dim=10)
    # eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(L)
    # compute the heat kernel
    heat_kernel = eigvecs @ torch.diag(torch.exp(-t * eigvals)) @ eigvecs.T
    heat_kernel = (heat_kernel + heat_kernel.T) / 2
    heat_kernel[heat_kernel < 0] = 0.0
    return heat_kernel


@pytest.mark.parametrize("t", [0.1, 1.0,])
@pytest.mark.parametrize("order", [10, 30, 50])
@pytest.mark.parametrize("k", [10, 20])
def test_heat_kernel_knn(t, order, k):
    tol = 2e-1 if t > 1.0 else 1e-1
    data = torch.randn(100, 5)
    heat_op = HeatKernelKNN(k=k, t=t, order=order, graph_type="scanpy")
    heat_kernel = heat_op(data)
    
    # test if symmetric
    assert torch.allclose(heat_kernel, heat_kernel.T)

    # test if positive
    assert torch.all(heat_kernel >= 0)

    # test if the heat kernel is close to the ground truth
    gt_heat_kernel = gt_heat_kernel_knn(data, t=t, k=k)
    assert torch.allclose(heat_kernel, gt_heat_kernel, atol=tol, rtol=tol)

if __name__ == "__main__":
    pytest.main([__file__])