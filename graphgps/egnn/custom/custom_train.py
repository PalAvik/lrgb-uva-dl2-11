import torch
import time
import logging

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, \
    clean_ckpt

from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
dtype = torch.float32

def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    
    for batch in loader:
     
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        
        nodes = batch["x"][:,:12].to(torch.device(cfg.device))
        positions = batch["x"][:,12:].to(torch.device(cfg.device))
        edges = batch["edge_index"].to(torch.device(cfg.device))
        edge_attr = batch["edge_attr"].to(torch.device(cfg.device))
        n_nodes = batch["x"].size(0)
        n_edges = edges.size(1)
        atom_mask = torch.ones(n_nodes,1).to(torch.device(cfg.device))
        edge_mask = torch.ones(n_edges,1).to(torch.device(cfg.device))
        
        true = batch["y"]
        pred = model(h0=nodes, x=positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))

        nodes = batch["x"][:,:12].to(torch.device(cfg.device))
        positions = batch["x"][:,12:].to(torch.device(cfg.device))
        edges = batch["edge_index"].to(torch.device(cfg.device))
        n_nodes = batch["x"].size(0)
        n_edges = edges.size(1)
        atom_mask = torch.ones(n_nodes,1).to(torch.device(cfg.device))
        edge_mask = torch.ones(n_edges,1).to(torch.device(cfg.device))

        true = batch["y"]
        pred = model(h0=nodes, x=positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


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
    print("Custom train started")
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
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):

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

        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
        
        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)
        
       
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))