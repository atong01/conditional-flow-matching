from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compare_single_graph_bayesian_shd(true_graph, estimated_graph):
    """Compute performance measures on encoded distribution over graphs
    Args:
        true_graph: (dxd) np.array, the true adjacency matrix, encoded in
            negative values are the nodes that could be one or zero.
        estimated graph: (dxd) np.array, the estimated adjacency matricies
            (weighted or unweighted) where b is the batch size
    """

    def shd(a, b):
        return np.sum(np.abs(a - b))

    true_graph = true_graph.squeeze().astype(int)
    var_maps = np.minimum(0, true_graph)[:, 0]
    var_mask = var_maps < 0
    vars_to_deidentify = -(var_maps[var_mask] + 1)
    estimated_graph = estimated_graph.squeeze()
    summed_estimated_graph = estimated_graph[~var_mask]
    # Distance to the nearest admissible graph.
    for i, v in enumerate(vars_to_deidentify):
        summed_estimated_graph[v] += estimated_graph[var_mask][i]
    hamming = shd(true_graph[~var_mask], summed_estimated_graph)
    return hamming


def compare_graphs_bayesian_shd(true_graph, estimated_graphs):
    shd = np.mean(
        [compare_single_graph_bayesian_shd(true_graph, graph) for graph in estimated_graphs]
    )
    thresholded_shd = np.mean(
        [
            compare_single_graph_bayesian_shd(true_graph, (graph > 0.5).astype(float))
            for graph in estimated_graphs
        ]
    )
    return shd, thresholded_shd


def compare_graphs_bayesian_dist(true_graph, estimated_graphs):
    """Compute performance measures on encoded distribution over graphs
    Args:
        true_graph: (dxd) np.array, the true adjacency matrix, encoded in
            negative values are the nodes that could be one or zero.
        estimated graph: (dxd) np.array, the estimated adjacency matricies
            (weighted or unweighted) where b is the batch size
    """

    def shd(a, b):
        return np.sum(np.abs(a - b))

    true_graph = true_graph.squeeze().astype(int)
    var_maps = np.minimum(0, true_graph)[:, 0]
    var_mask = var_maps < 0
    vars_to_deidentify = -(var_maps[var_mask] + 1)
    unique, counts = np.unique(vars_to_deidentify, return_counts=True)
    admissible_count = Counter()
    sample_count = Counter()
    for estimated_graph in estimated_graphs:
        summed_estimated_graph = estimated_graph[~var_mask]
        # Distance to the nearest admissible graph.
        for i, v in enumerate(vars_to_deidentify):
            summed_estimated_graph[v] += estimated_graph[var_mask][i]
        hamming = shd(true_graph[unique], summed_estimated_graph[unique])
        mask = var_mask.copy()
        mask[unique] = True
        sample_count.update([tuple(estimated_graph[mask].flatten())])
        if hamming == 0:
            admissible_count.update([tuple(estimated_graph[mask].flatten())])

    # Consider the undetermined edges only, lets count the # of unique admissible graphs observed?
    seen_admissible = len(list(admissible_count))
    unique_admissible = len(admissible_count)
    total_targets = np.sum(true_graph[unique], axis=1)
    total_admissible = 1
    for c, t in zip(counts, total_targets):
        total_admissible *= (c + 1) ** t

    return (
        seen_admissible,
        total_admissible,
        unique_admissible,
        admissible_count,
        sample_count,
    )


def compare_graphs_bayesian_cover(true_graph, estimated_graphs):
    (
        seen_admissible,
        total_admissible,
        unique_admissible,
        admissible_count,
        sample_count,
    ) = compare_graphs_bayesian_dist(true_graph, estimated_graphs)
    print("id-graphs:", unique_admissible, "-- total graphs:", total_admissible)
    return unique_admissible / total_admissible


def compute_gfn_neg_log_likelihood(true_graph, estimated_graphs, p_mse):
    r"""
    Warning: Currently Not being used.


    NNL = - \sum_G p(G | D)P(D | G)

    G - possible graphs in search space
    P(D | G) - given a graph, we can calculate the MSE of the data.
    P(G | D) - the probability of generating a graph given the data. This
               is generate using the learned P_F(G)
             - Need to compute P(G) over possible trajectories
    """
    pass


def compare_graph_distribution(true_graph, estimated_graphs):
    (
        seen_admissible,
        total_admissible,
        unique_admissible,
        admissible_count,
        sample_count,
    ) = compare_graphs_bayesian_dist(true_graph, estimated_graphs)
    # compute distacne to uniform
    dist_admissible = [
        float(x) / float(sum(admissible_count.values())) for x in list(admissible_count.values())
    ]
    entropy_admissible = 0.0
    for p in dist_admissible:
        if p == 0.0:
            entropy_admissible += 0.0
        else:
            entropy_admissible += p * np.log2(p)
    kl_unif = np.log2(len(admissible_count)) - entropy_admissible

    # compute proportion of admissible graphs
    admissible_proportion = [
        float(x) / float(sum(sample_count.values())) for x in list(admissible_count.values())
    ]

    # graph variation dist
    entropy_proportion = 0.0
    for p in admissible_proportion:
        if p == 0.0:
            entropy_proportion += 0.0
        else:
            entropy_proportion += p * np.log2(p)
    kl_proportion = np.log2(len(sample_count)) - entropy_proportion

    return kl_unif, admissible_proportion, kl_proportion


def compute_graphs_bayesian_diversity(graphs):
    """
    Input(s):
        - n_ens graphs: [n_ens, d, d]
    Output:
        - diversity metric: node-wise variance of predicted graphs
        normalized by node-wise varaince of graph generated with
        Bernoulli random variable.
    """
    ber_graphs = np.random.binomial(1, 0.5, size=graphs.shape)
    node_wise_var = np.var(graphs, axis=0)
    diversity = np.sum(node_wise_var)
    return diversity / np.sum(np.var(ber_graphs, axis=0))


def compute_graphs_sparsity(graph):
    """
    Input(s):
        - n_ens graphs: [n_ens, d, d]
    Output:
        - average sparsity metric
    """
    Adj = np.around(graph, decimals=0)
    sparsity = 1 - np.mean(Adj)
    return sparsity


def compare_graphs(true_graph, estimated_graph):
    """Compute performance measures on (binary) adjacency matrix.

    Input:
     - true_graph: (dxd) np.array, the true adjacency matrix
     - estimated graph: (dxd) np.array, the estimated adjacency matrix (weighted or unweighted)
    """
    # Handle new case where we encode information in the negative numbers
    true_graph = np.maximum(0, true_graph)

    def structural_hamming_distance(W_true, W_est):
        """Computes the structural hamming distance."""
        pred = np.flatnonzero(W_est != 0)
        cond = np.flatnonzero(W_true)
        cond_reversed = np.flatnonzero(W_true.T)
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        pred_lower = np.flatnonzero(np.tril(W_est + W_est.T))
        cond_lower = np.flatnonzero(np.tril(W_true + W_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)
        return shd

    num_edges = len(true_graph[np.where(true_graph != 0.0)])

    tam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in true_graph])
    eam = np.array([[1 if x != 0.0 else 0.0 for x in y] for y in estimated_graph])

    tp = len(np.argwhere((tam + eam) == 2))
    fp = len(np.argwhere((tam - eam) < 0))
    tn = len(np.argwhere((tam + eam) == 0))
    fn = num_edges - tp
    x = [tp, fp, tn, fn]

    if x[0] + x[1] == 0:
        precision = 0
    else:
        precision = float(x[0]) / float(x[0] + x[1])
    if tp + fn == 0:
        tpr = 0
    else:
        tpr = float(tp) / float(tp + fn)
    if x[2] + x[1] == 0:
        specificity = 0
    else:
        specificity = float(x[2]) / float(x[2] + x[1])
    if precision + tpr == 0:
        f1 = 0
    else:
        f1 = 2 * precision * tpr / (precision + tpr)
    if fp + tp == 0:
        fdr = 0
    else:
        fdr = float(fp) / (float(fp) + float(tp))

    shd = float(structural_hamming_distance(true_graph, estimated_graph))
    thresh_shd = float(
        structural_hamming_distance(true_graph, (estimated_graph > 0.5).astype(float))
    )

    if np.all(true_graph.flatten()):
        AUC = -1
        AP = -1
    else:
        AUC = roc_auc_score(true_graph.flatten(), estimated_graph.flatten())
        AP = average_precision_score(true_graph.flatten(), estimated_graph.flatten())

    metrics = ["tpr", "fdr", "shd", "tshd", "auc", "ap", "f1", "specificity"]
    values = [tpr, fdr, shd, thresh_shd, AUC, AP, f1, specificity]
    return dict(zip(metrics, values))
