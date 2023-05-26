import numpy as np
import torch
import time
import logging
import numpy as np
import os
import glob
import os.path as osp
from typing import  List, Union
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, \
    clean_ckpt
from e3nn.o3 import Irreps
from e3nn.o3 import Irreps, spherical_harmonics
from torch_geometric.data import Data
from torch_geometric.graphgym.register import register_train

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
dtype = torch.float32
from torch_scatter import scatter

def get_ckpt_epochs() -> List[int]:
    paths = glob.glob(get_ckpt_path('*'))
    return sorted([int(osp.basename(path).split('.')[0]) for path in paths])


def get_ckpt_path(epoch: Union[int, str]) -> str:
    return osp.join(get_ckpt_dir(), f'{epoch}.ckpt')

def get_ckpt_dir() -> str:
    return osp.join(cfg.run_dir, 'ckpt')

def clean_ckpt(best_epoch):
    r"""Removes all but the last model checkpoint."""
    for epoch in get_ckpt_epochs()[:-1]:
        if epoch != best_epoch:
          os.remove(get_ckpt_path(epoch))
 
class O3Transform:
    def __init__(self, lmax_attr):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    def __call__(self, graph):
        pos = graph.pos #batch X 3
#         vel = graph.vel
#         charges = graph.charges

#         prod_charges = charges[graph.edge_index[0]] * charges[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0].long()] - pos[graph.edge_index[1].long()]
#         edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='integral')
#         vel_embedding = spherical_harmonics(self.attr_irreps, vel, normalize=True, normalization='integral')
        dm = scatter(graph.edge_attr, graph.edge_index[1].to(torch.int64), dim=0, reduce="mean") 
        empty=torch.zeros(graph.x.shape[0],16).to(cfg.device)
        empty[0:dm.shape[0],:]=dm


        graph.node_attr=empty
        # print("node_attr",graph.node_attr.shape,graph.edge_attr.shape,graph.edge_index[1].shape)


#         vel_abs = torch.sqrt(vel.pow(2).sum(1, keepdims=True))
#         mean_pos = pos.mean(1, keepdims=True)

        #torch.cat((pos - mean_pos, vel, vel_abs), 1)
        graph.additional_message_features=None
#         graph.additional_message_features = torch.cat((edge_dist, prod_charges), dim=-1)
        return graph





def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    total_loss=0
    for batch in loader:
     
        batch.split = 'train'
        optimizer.zero_grad()
        extra=torch.zeros(batch["x"].shape[0],1)
        batch["x"]=torch.cat([batch["x"],extra],1)
        # print("SIZEEEEEE",batch["x"].shape,batch.num_graphs)
        batch.to(torch.device(cfg.device))

        
        nodes = batch["x"][:,:].to(torch.device(cfg.device))
        positions = batch["x"][:,12:].to(torch.device(cfg.device))
        edges = batch["edge_index"].to(torch.device(cfg.device))
        edge_attr = batch["edge_attr"].to(torch.device(cfg.device))
        n_nodes = batch["x"][:,:].size(0)
        n_edges = edges.size(1)
  
        true = batch["y"]
        batch_size=batch.num_graphs
        
        
        
        graph_Z=Data(x=nodes,edge_index=edges, pos=positions,  y=true)
        # graph_Z.x=nodes
        
        batchz = torch.arange(0, 1)
        graph_Z.batch = batchz.repeat_interleave(n_nodes).long()
        # print("VVVV",graph_Z.batch.shape,n_nodes,batch_size,len(loader))
        
        transform = O3Transform(3)
        
        graph_Z = transform(graph_Z)
        
        pred = model(graph_Z.to(torch.device(cfg.device)))
        # print(pred.shape,true.shape,"TRUTH")
#         pred = model(h0=nodes, x=positions, edges=edges, edge_attr=edge_attr)
        loss, pred_score = compute_loss(pred, true)
        total_loss=total_loss+loss
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
        # break
    print("total_loss=",total_loss)
    # scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        extra=torch.zeros(batch["x"].shape[0],1)
        batch["x"]=torch.cat([batch["x"],extra],1)
        batch.to(torch.device(cfg.device))
        

        nodes = batch["x"][:,:].to(torch.device(cfg.device))
        positions = batch["x"][:,12:].to(torch.device(cfg.device))
        edges = batch["edge_index"].to(torch.device(cfg.device))
        edge_attr = batch["edge_attr"].to(torch.device(cfg.device))
        n_nodes = batch["x"].size(0)
        n_edges = edges.size(1)
        batch_size=nodes.shape[0]

        true = batch["y"]
        
        graph_Z=Data(edge_index=edges, pos=positions,x=nodes,  y=true)
        graph_Z.x=nodes
        
        batchz = torch.arange(0, 1)
        graph_Z.batch = batchz.repeat_interleave(n_nodes).long()
        transform = O3Transform(3)
        
        graph_Z = transform(graph_Z)
        pred = model(graph_Z.to(torch.device(cfg.device)))
        
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
        # break


def train(loggers, loaders, model, optimizer, scheduler):
    """
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))
    
    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        stats = loggers[0].write_epoch(cur_epoch)
        perf[0].append(stats)

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                stats = loggers[i].write_epoch(cur_epoch)
                perf[i].append(stats)
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        
        full_epoch_times.append(time.perf_counter() - start_time)
        if is_ckpt_epoch(cur_epoch) or True: # saves for every epoch
            save_ckpt(model, optimizer, scheduler, cur_epoch)
        
        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)
        
                # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                print("found the best epoch")
                clean_ckpt(best_epoch)
                print("removed ckpt files except for epoch :{} ".format(best_epoch))
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
        
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")  

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean and False:
        clean_ckpt()
    
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))