import torch
import torch_geometric as tg
import networkx as nx
import numpy as np
import pandas as pd

from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels
from functools import cached_property


class NoiserHelper:

    def __init__(self, dataset):
        self.dataset = dataset

    @cached_property
    def mean_of_means(self):
        """ Gives mean input feature values across the entire graph"""
        mean_of_means = []
        for d in self.dataset:
            graph_mean = d.x.mean(dim=0)
            mean_of_means.append(graph_mean)
        mean_of_means = torch.row_stack(mean_of_means)
        mean_of_means = mean_of_means.mean(0)

        return mean_of_means

    @staticmethod
    def standard_deviation_of_input_features(data: tg.data.Data):
        standard_deviation_of_input = data.x.std(dim=0)
        return standard_deviation_of_input

    @staticmethod
    def mean_of_input_features(data: tg.data.Data):
        mean_of_input = data.x.mean(dim=0)
        return mean_of_input


class OneGraphNoise:

    def __init__(self, data: tg.data.Data, model: torch.nn.Module):
        self.data = data
        self.graph = tg.utils.convert.to_networkx(data)
        self.model = model

    @cached_property
    def all_shortest_paths(self):
        all_shortest_paths = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(self.graph)
        return all_shortest_paths.astype(int)

    def get_path_length_buckets(self, target_node_label):
        # get the shortest paths to target node
        shortest_paths_to_target = self.all_shortest_paths[:, target_node_label]  # [i, j] is the shortest path from i to j

        shortest_paths_df = pd.DataFrame(shortest_paths_to_target)

        renamer = {'index': 'node_id',
                   0: 'path_length'}
        shortest_paths_df = shortest_paths_df.reset_index().rename(columns=renamer)

        # Obtain the buckets corresponding to each path length. A map path_length -> node_ids with that path length
        path_length_buckets = shortest_paths_df.groupby('path_length')['node_id'].groups

        return path_length_buckets

    def obtain_modified_data(self,
                             path_length_buckets: dict,
                             path_length: int,
                             replacement_value: torch.Tensor,
                             down_sampling=None
                             ):

        # TODO add in random sampling of the path length buckts
        if replacement_value:
            new_data = self.data.clone()
            indices = path_length_buckets[path_length]
            if down_sampling:
                assert type(down_sampling) == float
                down_sampling_size = int(down_sampling*len(indices))
                indices = np.random.choice(indices, replace=False, size=down_sampling_size)

            for index in indices:
                new_data.x[index, :] = replacement_value # can swap this bit out based on what we want to do

            return new_data
        else:
            return self.data



    def get_result_for_all_path_lengths(self, target_node_label, replacement_value):
        """Returns modified """
        buckets = self.get_path_length_buckets(target_node_label)

        result = {}
        for path_length in buckets.keys():
            modified_data = self.obtain_modified_data(buckets,
                                                      path_length,
                                                      replacement_value=replacement_value)

            logits, label = self.model(modified_data)
            predictions = logits.argmax(dim=1)

            correct_prediction = (predictions == label)[target_node_label]
            result = path_length[correct_prediction]

        return result





