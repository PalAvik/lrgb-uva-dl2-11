"""
1. Place this script right where it is now. (In the same level as main.py)
2. In the config's yaml file you'll have to add a finetune parameter which points to the result
   folder of Pascal's GCN. (I'll send you the ss of how it should look like)
3. Call that yaml file when you run the script like this:
python model_inference.py --cfg configs/GCN/vocsuperpixels-GCN.yaml device cpu
4. Note that the predictions sent by model are not softmaxed and may look like this:
 (N X C) which is no of graphs by no of classes.
5. The softmax is taken inside the loss function. (let me know if you need more dets regarding this
)


"""

import os

import pandas as pd
import torch
import graphgps  # noqa, register custom modules

import torch_geometric as tg
import networkx as nx
import numpy as np
import pandas as pd
from functools import cached_property
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             )
from torch_geometric.graphgym.loader import create_loader

from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.loss import compute_loss
from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
from graphgps.custom.egnn import custom_egnn
from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels
from tqdm import tqdm
import pickle
from analysis.noising_experiments.noise_utils import get_predictions

from analysis.noising_experiments.noiser import NoiserHelper, OneGraphNoise



def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def dump_pkl(content, file_name):
    file = open(file_name, 'wb')
    pickle.dump(content, file)
    file.close()



if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    print('Loading Model')
    assert cfg.train.finetune

    cfg = load_pretrained_model_cfg(cfg)
    loggers = create_logger()
    loaders = create_loader()
    file_name = f"noising_exp_{cfg.model.type}.pkl"
    if cfg.model.type == 'egnn':
        model = custom_egnn.EGNN2(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                  device=cfg.device)
    elif cfg.model.type == 'enn':
        model = custom_egnn.EGNN(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                 device=cfg.device)
    else:
        model = create_model()
        # !! Nik update this line to the extra param of device on YOUR LOCAL BRANCH. !!
        model = init_model_from_pretrained(model, cfg.train.finetune,  cfg.train.freeze_pretrained) 
    
    print(model)
    dataset = VOCSuperpixels(root='datasets/VOCSuperpixels',
                             slic_compactness=10,
                             name='edge_wt_only_coord',
                             split='test')

    print('Dataset loaded')

    N = 20
    helper = NoiserHelper(dataset)
    results_per_graph = []
    for graph_id in range(N):
        data = dataset[graph_id]
        data = data.to(torch.device(cfg.device))
        data = dataset[graph_id]
        noiser = OneGraphNoise(data, model)
        result_new = noiser.get_results_for_all_target_nodes(replacement_value=helper.mean_of_means)
        
        predictions = get_predictions(data, model)


        out_frames = []
        for target_node in range(result_new.shape[0]):
            row = {'graph_id': [graph_id],
                    'target_node': [target_node],
                    'truth': [data.y[target_node].item()],
                    'standard_prediction': [predictions[target_node].item()]
                    }
            for path_length in range(result_new.shape[1]):
                row[f"path_length_{path_length}_prediction"] = [result_new[target_node, path_length]]
            row = pd.DataFrame.from_dict(row)
            out_frames.append(row)
        df = pd.concat(out_frames)
        print(results_per_graph)
        results_per_graph.append(df)
    
    dump_pkl(results_per_graph, file_name=file_name)
    print("Content dumped in pkl file: ", file_name)



