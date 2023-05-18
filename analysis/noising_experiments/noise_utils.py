import torch
from torch_geometric.graphgym.loss import compute_loss

from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             )

def get_predictions(data, model):
    if cfg.model.type == 'enn':
            nodes = data["x"][:,:12].to(torch.device(cfg.device))
            positions = data["x"][:,12:].to(torch.device(cfg.device))
            edges = data["edge_index"].to(torch.device(cfg.device))
            edge_attr = data["edge_attr"].to(torch.device(cfg.device))
            true = data["y"]
            pred = model(h0=nodes, x=positions, edges=edges, edge_attr=edge_attr)
    else:
        data = data.to(torch.device(cfg.device))
        pred, true = model(data)

    _ , pred_score = compute_loss(pred, true)
    predictions = pred_score.max(dim=1)[1]
    return predictions