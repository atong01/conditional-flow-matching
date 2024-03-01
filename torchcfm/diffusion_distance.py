import torch
from torchcfm.cheb_approx import compute_chebychev_coeff_all, expm_multiply

try:
    import scanpy as sc
except ImportError:
    pass

EPS_LOG = 1e-6
EPS_HEAT = 1e-4

def norm_sym_laplacian(A: torch.Tensor):
    deg = A.sum(dim=1)
    deg_sqrt_inv = torch.diag(1.0 / torch.sqrt(deg + EPS_LOG))
    return deg_sqrt_inv @ A @ deg_sqrt_inv


def torch_knn_from_data(
    data: torch.Tensor, k: int, projection: bool = False, proj_dim: int = 100
):
    if projection:
        _, _, V = torch.pca_lowrank(data, q=proj_dim, center=True)
        data = data @ V
    dist = torch.cdist(data, data)
    _, indices = torch.topk(dist, k, largest=False)
    affinity = torch.zeros(data.shape[0], data.shape[0])
    affinity.scatter_(1, indices, 1)
    return norm_sym_laplacian(affinity)


def scanpy_knn_from_data(
    data: torch.Tensor, k: int, projection: bool = False, proj_dim: int = 100
):
    adata = sc.AnnData(data.numpy())
    if projection:
        sc.pp.pca(adata, n_comps=proj_dim)
    sc.pp.neighbors(
        adata, n_neighbors=k, use_rep="X_pca" if projection else None
    )
    return norm_sym_laplacian(
        torch.tensor(adata.obsp["connectivities"].toarray())
    )


def var_fn(x, t):
    outer = torch.outer(torch.diag(x), torch.ones(x.shape[0]))
    vol_approx = (outer + outer.T) * 0.5
    return -t * torch.log(x + EPS_LOG) + t * torch.log(vol_approx + EPS_LOG)


class BaseHeatKernel:
    def __init__(self, t: float = 1.0, order: int = 30):
        self.t = t
        self.order = order
        self.dist_fn = var_fn
        self.graph_fn = None

    def __call__(self, data: torch.Tensor):
        if self.graph_fn is None:
            raise NotImplementedError("graph_fn is not implemented")
        L = self.graph_fn(data)
        heat_kernel = self.compute_heat_from_laplacian(L)
        heat_kernel = self.sym_clip(heat_kernel)
        return heat_kernel

    def compute_heat_from_laplacian(self, L: torch.Tensor):
        n = L.shape[0]
        val = torch.linalg.eigvals(L).real
        max_eigval = val.max()
        cheb_coeff = compute_chebychev_coeff_all(
            0.5 * max_eigval, self.t, self.order
        )
        heat_kernel = expm_multiply(
            L, torch.eye(n), cheb_coeff, 0.5 * max_eigval
        )
        return heat_kernel

    def sym_clip(self, heat_kernel: torch.Tensor):
        heat_kernel = (heat_kernel + heat_kernel.T) / 2
        heat_kernel[heat_kernel < 0] = 0.0 + EPS_HEAT
        return heat_kernel

    def fit(self, data: torch.Tensor, dist_type: str = "var"):
        assert dist_type in self.dist_fn
        heat_kernel = self(data)
        return self.dist_fn[dist_type](heat_kernel, self.t)
    

class HeatKernelKNN(BaseHeatKernel):
    """Approximation of the heat kernel with a graph from a k-nearest neighbors affinity matrix.
    Uses Chebyshev polynomial approximation.
    """

    _is_differentiable = False
    _implemented_graph = {
        "torch": torch_knn_from_data,
        "scanpy": scanpy_knn_from_data,
    }

    def __init__(
        self,
        k: int = 10,
        order: int = 30,
        t: float = 1.0,
        projection: bool = False,
        proj_dim: int = 100,
        graph_type: str = "scanpy",
    ):
        super().__init__(t=t, order=order)
        assert (
            graph_type in self._implemented_graph
        ), f"Type must be in {self._implemented_graph}"
        self.k = k
        self.projection = projection
        self.proj_dim = proj_dim
        self.graph_fn = lambda x: self._implemented_graph[graph_type](
            x, self.k, projection=self.projection, proj_dim=self.proj_dim
        )
