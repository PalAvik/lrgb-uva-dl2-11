import datetime
import os
import collections
import pickle
import torch
import logging
import copy
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

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger

from graphgps.metrices.cheeger_lower_bound_metric import LowerCheegerMetric
from graphgps.metrices.cheeger_upper_bound_metric import UpperCheegerMetric
from graphgps.metrices.diameter_metric import DiameterMetric
from graphgps.metrices.shortest_path_metric import ShortestPathMetric

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

path = "/home/lcur1702/lrgb_madhura/lrgb-uva-dl2-11/pascal_metrics.pkl"

file = open(path, 'rb')
data = pickle.load(file)
print(len(data))              
file.close()



if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    lc = LowerCheegerMetric()
    uc = UpperCheegerMetric()
    dm = DiameterMetric()
    sp = ShortestPathMetric()

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    if cfg.train.finetune:
        cfg = load_pretrained_model_cfg(cfg)
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        entries = []
        file_name = "pascal_metrics.pkl"
        if cfg.train.finetune: 
            model = init_model_from_pretrained(model, cfg.train.finetune,  cfg.train.freeze_pretrained)
            
            print(model)      
            # j = 0                          
            for batch in loaders[0]:
                batch.split = "test"
                # if j == 25: break
                # batch = batch.to(torch.device(cfg.device))
                for i in range(batch.num_graphs):
                     graph = batch[i]
                     lc_val =  lc.get_graph_value(graph, normalization="sym", is_undirected=True)
                     uc_val =  uc.get_graph_value(graph, normalization="sym", is_undirected=True)
                     dm_val = dm.get_graph_value(graph, normalization="sym", is_undirected=True)
                     s_val = sp.get_graph_value(graph, normalization="sym", is_undirected=True)
                     entries.append({"lc":lc_val,"uc":uc_val,"dm":dm_val,"sp":s_val})
                print("batch done")
                # j +=  1
            

            dump_pkl(entries, file_name=file_name)
            print("Content dumped in pkl file: ", file_name)
            
            

                
                
          