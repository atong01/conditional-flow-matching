import numpy as np
import scanpy as sc


def adata_dataset(path, embed_name="X_pca", label_name="day", max_dim=100):
    adata = sc.read_h5ad(path)
    labels = adata.obs[label_name].astype("category")
    ulabels = labels.cat.categories
    return adata.obsm[embed_name][:, :max_dim], labels, ulabels


def tnet_dataset(path, embed_name="pcs", label_name="sample_labels", max_dim=100):
    a = np.load(path, allow_pickle=True)
    return a[embed_name][:, :max_dim], a[label_name], np.unique(a[label_name])


def load_dataset(path, max_dim=100):
    if path.endswith("h5ad"):
        return adata_dataset(path, max_dim=max_dim)
    if path.endswith("npz"):
        return tnet_dataset(path, max_dim=max_dim)
    raise NotImplementedError()
