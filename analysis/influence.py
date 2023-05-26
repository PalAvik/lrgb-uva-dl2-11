import pickle
import torch
import torch_geometric as tg

import networkx as nx

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def row_normalise(inf):

    row_totals = inf.sum(axis=1)
    norm = (inf.T/row_totals).T

    return norm


def convert_positions_to_df(positions_array, label):
    assert label in {'source', 'target'}

    coords = pd.DataFrame(positions_array)

    coords.index.name = label
    coords = coords.reset_index()
    coords = coords.rename(columns={0: f'{label}_x',
                                    1: f'{label}_y'}
                           )
    return coords


def convert_arrays_to_df(influence_matrix, positions):
    df = pd.DataFrame(influence_matrix)

    df.index.name = 'source'  # TODO is this right?

    melted = df.melt(ignore_index=False)
    melted = melted.reset_index()  # Move source index to be a column, gives unique index
    melted = melted.rename(columns={'variable': 'target',
                                    'value': 'influence_score',
                                    'index': 'source'})

    source_coords = convert_positions_to_df(positions, label='source')
    target_coords = convert_positions_to_df(positions, label='target')

    melted = melted.merge(source_coords, on='source', how='left')
    melted = melted.merge(target_coords, on='target', how='left')

    return melted

def calculate_distance(influence_df):
    influence_df['distance_sq'] = (influence_df['source_x'] - influence_df['target_x'])**2 + (influence_df['source_y'] -
                                                                        influence_df['target_y'] )**2
    influence_df['distance'] = np.sqrt(influence_df['distance_sq'])

    return influence_df


def get_graph(edge_array, position_array):

    edges = edge_array.astype(int)

    edges = torch.Tensor(edges)
    position_array = torch.Tensor(position_array)

    data = tg.data.Data(edge_index=edges,
                        x=position_array)

    graph = tg.utils.convert.to_networkx(data)

    return graph


def add_graph_distances_to_df(influence_df, graph):
    shortest_paths = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(graph, weight=None)
    shortest_paths = pd.DataFrame(shortest_paths)

    shortest_paths_map = shortest_paths.melt(ignore_index=False)
    shortest_paths_map = shortest_paths_map.reset_index()
    shortest_paths_map = shortest_paths_map.rename(columns={'index': 'source',
                                                            'variable': 'target',
                                                            'value': 'graph_distance'}
                                                   )

    shortest_paths_map['graph_distance'] = shortest_paths_map['graph_distance'].astype('int')

    influence_df = influence_df.merge(shortest_paths_map, on=['source', 'target'])

    return influence_df

def raw_data_to_df(influence_matrix, position_array, edge_array, normalise):

    if normalise:
        influence_matrix = row_normalise(influence_matrix)
        assert np.allclose(influence_matrix.sum(axis=1), 1.0)

    influence_df = convert_arrays_to_df(influence_matrix, position_array)
    influence_df = calculate_distance(influence_df)
    graph = get_graph(edge_array, position_array)

    influence_df = add_graph_distances_to_df(influence_df, graph)

    return  influence_df


def process_all_graphs(pickle_file, normalise=False):
    with open(pickle_file, 'rb') as f:
        all_graphs = pickle.load(f)
        print(f"{len(all_graphs)} graphs loaded")

    dfs = []
    for i,graph in enumerate(all_graphs):
        try:
            influence_df = raw_data_to_df(graph['influence_score'],
                                          graph['xpos'],
                                          graph['edges'],
                                          normalise=normalise)
            influence_df['graph_id'] = i
            dfs.append(influence_df)
        except Exception as error: # bypass faulty graphs (if any)
            pass

    final_df = pd.concat(dfs)
    final_df = final_df.sort_values(by='influence_score')
    final_df = final_df.reset_index(drop=True)

    return final_df


def plot_mean_influence_by_distance(df, ax, label):
    per_distance_per_source = df.groupby(['graph_distance', 'target', 'graph_id'])['influence_score'].sum()
    expected_influence_per_distance = per_distance_per_source.groupby('graph_distance').mean()
    expected_influence_per_distance.plot(ax=ax, label=label, drawstyle='steps-mid')

