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
import argparse

import pandas as pd
import torch

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             )
from torch_geometric.graphgym.loader import create_loader

from torch_geometric.graphgym.model_builder import create_model

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
from graphgps.custom.egnn import custom_egnn
from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels

import pickle


from analysis.noising_experiments.noiser import OneGraphNoise, NoiserHelper


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

def noiser_parse_arg() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('--device', type=str, default='cpu', help='torch device')
    parser.add_argument('--num_graphs', type=int, default=1)
    parser.add_argument('--output_file', type=str, required=True)

    ### What is the point of this??
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')


    return parser.parse_args()



if __name__ == '__main__':
    # Load cmd line args
    args = noiser_parse_arg()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    cfg.device = args.device
    print('Loading Model')
    assert cfg.train.finetune

    cfg = load_pretrained_model_cfg(cfg)
    loggers = create_logger()
    loaders = create_loader()

    if cfg.model.type == 'egnn':
        model = custom_egnn.EGNN2(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                  device=args.device)
        is_graphgym = False
    elif cfg.model.type == 'enn':
        model = custom_egnn.EGNN(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                 device=args.device)
        is_graphgym = False

    else:
        model = create_model()
        is_graphgym = True

    model = init_model_from_pretrained(model,
                                        cfg.train.finetune,
                                        cfg.train.freeze_pretrained,
                                        device=args.device
                                       )

    model.eval()

    entries = []
    file_name = f"inf_scores_{cfg.model.type}.pkl"

    dataset = VOCSuperpixels(root='datasets/VOCSuperpixels',
                             slic_compactness=10,
                             name='edge_wt_only_coord',
                             split='test')

    print('Dataset loaded')


    with torch.no_grad():
        results_per_graph = []

        for graph_id in range(args.num_graphs):

            data = dataset[graph_id]

            helper = NoiserHelper(dataset)
            noiser = OneGraphNoise(data, model)

            result_new = noiser.get_results_for_all_target_nodes(replacement_value=helper.mean_of_means)

            ## Get result for vanilla model (i.e. not fudged)
            logits, target = model(data)
            predictions = logits.argmax(dim=1)

            # Write all data for this graph
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
            results_per_graph.append(df)

        final = pd.concat(results_per_graph)
        final.to_pickle(args.output_file)




