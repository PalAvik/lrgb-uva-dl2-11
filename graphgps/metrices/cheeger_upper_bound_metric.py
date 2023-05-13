from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigs, eigsh


class UpperCheegerMetric:
    def _compute_lambda_1(self, graph, normalization="sym", is_undirected=True):
        # Compute the smallest non-zero eigenvalue of the normalized Laplacian matrix
        # :param graph: torch_geometric.data.Data object
        # :return: lambda_1
        # The normalized Laplacian matrix is defined as:
        # L = I - D^(-1/2) * A * D^(-1/2)
        edge_weight = graph.edge_attr
        # If edge weights are not specified, set them to 1
        if edge_weight is not None and edge_weight.numel() != graph.num_edges:
            edge_weight = None

        # Compute the normalized Laplacian matrix
        edge_index, edge_weight = get_laplacian(
            graph.edge_index,
            edge_weight,
            normalization=normalization,
            num_nodes=graph.num_nodes,
        )
        # Convert to scipy sparse matrix
        L = to_scipy_sparse_matrix(edge_index, edge_weight, graph.num_nodes)
        # Compute the eigenvalues
        # If the graph is undirected, the eigenvalues are real and the computation is faster, so use eigsh
        if is_undirected and normalization != "rw":
            eig_fn = eigsh
        # If the graph is directed, the eigenvalues are complex and the computation is slower, so use eigs
        else:
            eig_fn = eigs
        # Compute the eigenvalues. 'SM' means smallest magnitude
        eigenvalues = eig_fn(L.toarray(), k=4, which="SM", return_eigenvectors=False)
        # Return the smallest non-zero eigenvalue
        return eigenvalues[0]

    def get_graph_value(self, graph, normalization="sym", is_undirected=True):
        # Compute the metric for a single graph
        # :param graph: torch_geometric.data.Data object
        # :return: cheeger score
        # Cheeger score can be computed as:
        # h^2(G) ≈ 4 * lambda_1
        # lambda_1 is the smallest non-zero eigenvalue of the normalized Laplacian matrix
        lambda_1 = self._compute_lambda_1(
            graph, normalization=normalization, is_undirected=is_undirected
        )
        return 2 * lambda_1

    def __call__(self, graphs, normalization="sym", is_undirected=True):
        # Compute the metric for a batch of graphs, output a list of metric values
        # :param graphs: a torch_geometric.data.Batch of torch_geometric.data.Data objects
        # :param normalization: normalization method for the Laplacian matrix
        # :param is_undirected: whether the graph is undirected
        # :return: a list of cheeger scores
        # Compute the metric for each graph in the batch
        upper_values = []
        for i in range(graphs.num_graphs):
            graph = graphs[i]
            upper_value = self.get_graph_value(
                graph, normalization=normalization, is_undirected=is_undirected
            )
            upper_values.append(upper_value)
        return upper_values
