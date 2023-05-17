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
import torch

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_agg_dir, set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader

from torch_geometric.graphgym.model_builder import create_model

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
from graphgps.custom.egnn import custom_egnn
from graphgps.jacobian.utils import jacobian_graph
from graphgps.jacobian.graphutils import get_adj_matrix  # use this @Avik
from graphgps.transform.posenc_stats import compute_posenc_stats
import pickle
from tqdm import tqdm


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

    if cfg.train.finetune:
        cfg = load_pretrained_model_cfg(cfg)
        loggers = create_logger()
        loaders = create_loader()

        if cfg.model.type == 'egnn':
            model = custom_egnn.EGNN2(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                      device=cfg.device)
            is_graphgym = False
        elif cfg.model.type == 'enn':
            model = custom_egnn.EGNN(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                     device=cfg.device)
            is_graphgym = False

        else:
            model = create_model()
            is_graphgym = True

    model = init_model_from_pretrained(model,
                                       cfg.train.finetune,
                                       cfg.train.freeze_pretrained)

    model.eval()

    entries = []
    file_name = f"inf_scores_{cfg.model.type}.pkl"

    no_batches = 2
    data_path_dir = './datasets/VOCSuperpixels/small_test_set'
    uses_pe = False

    with torch.no_grad():
        for b_idx in tqdm(range(no_batches)):
            data_path = os.path.join(data_path_dir, f'batch_{b_idx}.pt')
            graph_batch = torch.load(data_path)

            for g_idx in tqdm(range(graph_batch.num_graphs)):

                data = graph_batch[g_idx]

                if cfg.posenc_LapPE.enable == True:
                    graph = compute_posenc_stats(graph, ['LapPE'], is_undirected=True, cfg=cfg)
                    uses_pe = True

                if cfg.model.type in ['enn', 'egnn']:
                    nodes = graph.x[:, :12].to(torch.device(cfg.device))
                else:
                    nodes = graph.x.to(torch.device(cfg.device))

                positions = graph.x[:, 12:].to(torch.device(cfg.device))
                edges = graph.edge_index.to(torch.device(cfg.device))
                edge_attr = graph.edge_attr.to(torch.device(cfg.device))

                true = graph.y
                nodes.requires_grad_(True)
                edges = edges.float()
                edges.requires_grad_(False)
                edge_attr.requires_grad_(True)

                if cfg.model.type in ['enn', 'egnn']:
                    input_ = (nodes, positions, edges, edge_attr)
                elif cfg.model.type == 'GPSModel':
                    EigVals = graph.EigVals.to(torch.device(cfg.device))
                    EigVecs = graph.EigVecs.to(torch.device(cfg.device))
                    input_ = (nodes, edges, edge_attr, EigVals, EigVecs)
                else:
                    input_ = (nodes, edges, edge_attr)

        dump_pkl(entries, file_name=file_name)
        print("Content dumped in pkl file: ", file_name)
