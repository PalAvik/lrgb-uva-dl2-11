from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigs, eigsh
import networkx as nx
import torch_geometric


class ShortestPathMetric:
    def get_graph_value(self, graph, normalization="sym", is_undirected=True):
        # Compute the metric for a single graph
        # :param graph: torch_geometric.data.Data object
        # :return: cheeger score
        g_nx = torch_geometric.utils.to_networkx(graph)
        g_nx = g_nx.to_undirected()
        return nx.average_shortest_path_length(g_nx)

    def __call__(self, graphs, normalization="sym", is_undirected=True):
        # Compute the metric for a batch of graphs, output a list of metric values
        # :param graphs: a torch_geometric.data.Batch of torch_geometric.data.Data objects
        # :param normalization: normalization method for the Laplacian matrix
        # :param is_undirected: whether the graph is undirected
        # :return: a list of cheeger scores
        # Compute the metric for each graph in the batch
        values = []
        for i in range(graphs.num_graphs):
            graph = graphs[i]
            value = self.get_graph_value(
                graph, normalization=normalization, is_undirected=is_undirected
            )
            values.append(value)
        return values
