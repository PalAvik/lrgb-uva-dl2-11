
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    print(adj.shape, sp.eye(adj.shape[0]).shape)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_adj_matrix(edge_index):
        edgeList = np.array(edge_index.transpose(1, 0))
        edgeList = list(map(tuple, edgeList))
        d = defaultdict(list)
        for k, v in edgeList:
            d[k].append(v)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(d))
        adj_norm = preprocess_graph(adj)
        return adj_norm
