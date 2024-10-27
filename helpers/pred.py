import numpy as np
import torch
from tqdm.auto import tqdm
import networkx as nx
import time
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from helpers import sinkhorn

def feature_extraction(G):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)

def eucledian_dist(F1, F2, n):
    D = euclidean_distances(F1, F2)
    return D

def dist(A, B, P):
    obj = np.linalg.norm(np.dot(A, P) - np.dot(P, B))
    return obj*obj/2

def FindQuasiPerm(A, B, D, mu, niter):
    n = len(A)
    P = torch.ones((n,n), dtype = torch.float64) / n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K = mu * D
    for i in range(niter):
        for it in range(1, 11):
            G = -torch.mm(torch.mm(A.T, P), B) - torch.mm(torch.mm(A, P), B.T) + K + i*(mat_ones - 2*P)
            q = sinkhorn.sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P

def convertToPermHungarian(M, n1, n2):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)

    P = np.zeros((n, n))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans

def fugal(Gq, Gt, mu, niter):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)
    for i in range(n1, n):
        Gq.add_node(i)
    for i in range(n2, n):
        Gt.add_node(i)

    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    D = eucledian_dist(F1, F2, n)
    D = torch.tensor(D, dtype = torch.float64)

    P = FindQuasiPerm(A, B, D, mu, niter)
    P_perm, ans = convertToPermHungarian(P, n1, n2)
    return ans

def predict_alignment(queries, targets, mu = 1, niter = 15):
    n = len(queries)
    mapping = []
    times = []
    for i in tqdm(range(n)):
        t1 = time.time()
        ans = fugal(queries[i], targets[i], mu, niter)
        mapping.append(ans)
        t2 = time.time()
        times.append(t2 - t1)
    return mapping, times