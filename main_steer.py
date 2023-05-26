import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_agg_dir, set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optimizer import create_optimizer, \
    create_scheduler, OptimizerConfig, SchedulerConfig
# from torch_geometric.graphgym.model_builder import create_model
# from torch_geometric.graphgym.train import train
# from graphgps.train import custom_train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
# from graphgps.custom.train_newschedule import train
from graphgps.custom import custom_train3
from graphgps.custom.segnn.segnn import SEGNN
from e3nn.o3 import Irreps, spherical_harmonics
from graphgps.custom.segnn.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return SchedulerConfig(scheduler=cfg.optim.scheduler,
                           steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
                           max_epoch=cfg.optim.max_epoch)


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


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

def create_model():
    
    
    task = "node"
    hidden_features=128
    lmax_h=2
    lmax_attr=3
    norm='instance'
    pool='avg'
    layers=4
    input_irreps = Irreps("12x0e+1x1o")
    output_irreps = Irreps("21x0e")
    edge_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
    node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
    subspace_type="weightbalanced"
    if subspace_type == "weightbalanced":
        hidden_irreps = WeightBalancedIrreps(
            Irreps("{}x0e".format(hidden_features)), node_attr_irreps, sh=True, lmax=lmax_h)
    elif subspace_type == "balanced":
        hidden_irreps = BalancedIrreps(lmax_h,hidden_features, True)
    else:
        raise Exception("Subspace type not found")
    additional_message_irreps = None#Irreps("2x0e")

    model = SEGNN(input_irreps,
                  hidden_irreps,
                  output_irreps,
                  edge_attr_irreps,
                  node_attr_irreps,
                  num_layers=layers,
                  norm=norm,
                  pool=pool,
                  task=task,
                  additional_message_irreps=additional_message_irreps).to(torch.device(cfg.device))
    return(model)




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
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.train.finetune:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        # print(model)
#         break
        if cfg.train.finetune: 
            model = init_model_from_pretrained(model, cfg.train.finetune,
                                               cfg.train.freeze_pretrained)
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        # print("heeeeeelp")
#         
        # Start training
#         if cfg.train.mode == 'standard':
#             if cfg.wandb.use:
#                 logging.warning("[W] WandB logging is not supported with the "
#                                 "default train.mode, set it to `custom`")
            # train(loggers, loaders, model, optimizer, scheduler)
        custom_train3.train(loggers, loaders, model, optimizer, scheduler)
#         else:
#             print("wrong")
#             train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
#                                        scheduler)
    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))
    logging.info(f"[*] All done: {datetime.datetime.now()}")
