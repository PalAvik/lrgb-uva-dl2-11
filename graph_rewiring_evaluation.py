import datetime
import os
import torch
import logging
import time

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_agg_dir, set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optimizer import create_optimizer, \
    create_scheduler, OptimizerConfig, SchedulerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
from torch_geometric.graphgym.loss import compute_loss

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
from graphgps.custom.egnn import custom_egnn
from graphgps.jacobian.utils import jacobian_graph
from graphgps.transform.posenc_stats import compute_posenc_stats
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F
from graphgps.train.custom_train import eval_epoch
from graphgps.sdrf.sdrf_utils import rewire_graph


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def _get_pred_int(pred_score):
    if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
        return (pred_score > cfg.model.thresh).long()
    else:
        return pred_score.max(dim=1)[1]


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
    set_printing()
    
    if cfg.train.finetune:
        cfg = load_pretrained_model_cfg(cfg)
        
        loaders = create_loader()
        loggers = create_logger()
        
        if cfg.model.type == 'egnn':
            model = custom_egnn.EGNN(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,device=cfg.device)
            is_graphgym = False
        elif cfg.model.type == 'enn':
            model = custom_egnn.EGNN(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                     device=cfg.device)
            is_graphgym = False
        else:
            model = create_model()
            is_graphgym = True
        
        model = init_model_from_pretrained(model, cfg.train.finetune,
                                               cfg.train.freeze_pretrained)
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        
        model.eval()
        first = True
        rewire = True
        new_edges_ratio = 0.01
        
        for new_edges_ratio in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        
            for batch in tqdm(loaders[2]):
                batch.split = "test"

                for g_idx in range(batch.num_graphs):
                    graph = batch[g_idx]

                    if cfg.posenc_LapPE.enable == True:
                        graph = compute_posenc_stats(graph, ['LapPE'], is_undirected=True, cfg=cfg)

                    if rewire:
                        if cfg.model.type == 'GPSModel':
                            EigVals = graph.EigVals
                            EigVecs = graph.EigVecs
                            graph = rewire_graph(graph, new_edges_ratio)
                            graph.EigVals = EigVals
                            graph.EigVecs = EigVecs
                        else:
                            graph = rewire_graph(graph, new_edges_ratio)

                    
                    if cfg.model.type in ['enn', 'egnn']:
                        nodes = graph.x[:, :12].to(torch.device(cfg.device))
                        positions = graph.x[:, 12:].to(torch.device(cfg.device))
                        edges = graph.edge_index.to(torch.device(cfg.device))
                        edge_attr = graph.edge_attr.to(torch.device(cfg.device))
                        
                        if cfg.model.type == 'enn':
                            input_ = (nodes, positions, edges, edge_attr)
                        elif cfg.model.type == 'egnn':
                            n_nodes_arr = [graph.x.size(0)]
                            tensor_n_nodes = torch.tensor(n_nodes_arr)
                            t_n = tensor_n_nodes.repeat_interleave(tensor_n_nodes).unsqueeze(1).to(torch.device(cfg.device))
                            input_ = (nodes, positions, edges, edge_attr, t_n.float())
                            
                        pred = model(*input_)
                        true = graph.y
                    
                    else:
                        pred, true = model(graph.to(torch.device(cfg.device)))
                    
                    pred = F.log_softmax(pred, dim=-1)
                    pred = _get_pred_int(pred)

                    if first:
                        first = False
                        all_preds = pred.cpu().detach().numpy()
                        all_true = true.cpu().detach().numpy()
                    else:
                        all_preds = np.concatenate((all_preds, pred.cpu().detach().numpy()))
                        all_true = np.concatenate((all_true, true.cpu().detach().numpy()))

#             print("ALL PREDS SHAPE: ", all_preds.shape)
#             print("ALL TRUE SHAPE: ", all_preds.shape)
            final_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
            print(f"FINAL F1 SCORE for {new_edges_ratio}: {final_f1}", final_f1)