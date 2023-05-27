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
from graphgps.custom.egnn import custom_egnn
from graphgps.jacobian.utils import jacobian_graph
from graphgps.jacobian.graphutils import get_adj_matrix  # use this @Avik
from graphgps.transform.posenc_stats import compute_posenc_stats
import pickle
from tqdm import tqdm
from graphgps.custom.segnn.segnn import SEGNN
from e3nn.o3 import Irreps, spherical_harmonics
from graphgps.custom.segnn.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from graphgps.custom.custom_train3 import O3Transform
from torch_geometric.data import Data

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


def get_influence_score(node_jacobian, adj_mat, positions):
    total_nodes = node_jacobian.size(0)
    inf_score = torch.zeros((total_nodes, total_nodes))
    distances = torch.zeros((total_nodes, total_nodes))

    node_jacobian = node_jacobian.sum((1, 3))
    node_jacobian = node_jacobian @ adj_mat

    for source in range(total_nodes):
        source_pos = positions[source].unsqueeze(0)
        for target in range(total_nodes):
            if source != target:
                h_x_y = node_jacobian[target, source]
                h_x_all = node_jacobian[:, source].sum()
                I_x_y = h_x_y / h_x_all
                inf_score[source][target] = I_x_y.abs().item()
                D_x_y = torch.cdist(source_pos, positions[target].unsqueeze(0), p=2)
                distances[source][target] = D_x_y.item()
    return inf_score, distances


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
        is_steer = False
        if cfg.model.type == 'egnn':
            model = custom_egnn.EGNN2(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                      device=cfg.device)
            is_graphgym = False
        elif cfg.model.type == 'enn':
            model = custom_egnn.EGNN(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=4, coords_weight=1.0,
                                     device=cfg.device)
            is_graphgym = False
        elif cfg.model.type == 'scgnn':


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


            model = SEGNN(
                input_irreps,
                hidden_irreps,
                output_irreps,
                edge_attr_irreps,
                node_attr_irreps,
                num_layers=layers,
                norm=norm,
                pool=pool,
                task=task,
                additional_message_irreps=additional_message_irreps,
                computing_jacobian=True,
                ).to(torch.device(cfg.device))
            is_graphgym = False            
        else:
            model = create_model()
            is_graphgym = True

        model = init_model_from_pretrained(model, cfg.train.finetune,
                                           cfg.train.freeze_pretrained)

        model.eval()

        entries = []
        file_name = f"inf_scores_{cfg.model.type}_coco.pkl"

        no_batches = 1
        data_path_dir = './datasets/COCOSuperpixels/small_test_set'
        uses_pe = False

        with torch.no_grad():
            for b_idx in tqdm(range(no_batches)):
                data_path = os.path.join(data_path_dir, f'batch_{b_idx}.pt')
                graph_batch = torch.load(data_path)

                for g_idx in tqdm(range(graph_batch.num_graphs)):

                    graph = graph_batch[g_idx]

                    if cfg.posenc_LapPE.enable == True:
                        graph = compute_posenc_stats(graph, ['LapPE'], is_undirected=True, cfg=cfg)
                        uses_pe = True

                    if cfg.model.type in ['enn', 'egnn']:
                        nodes = graph.x[:, :12].to(torch.device(cfg.device))
                    elif cfg.model.type in ['scgnn']:
                        nodes_d = graph.x.to(torch.device(cfg.device))
                        extra=torch.zeros(nodes_d.shape[0],1).to(torch.device(cfg.device))
                        nodes=torch.cat([nodes_d,extra],1)

                    else:
                        nodes = graph.x.to(torch.device(cfg.device))

                    if cfg.model.type in ['scgnn']:
                        positions=nodes[:,12:].to(torch.device(cfg.device))
                    else:
                        positions = graph.x[:, 12:].to(torch.device(cfg.device))


                    edges = graph.edge_index.to(torch.device(cfg.device))
                    edge_attr = graph.edge_attr.to(torch.device(cfg.device))

                    true = graph.y
                    nodes.requires_grad_(True)
                    edges = edges.float()
                    edges.requires_grad_(False)
                    edge_attr.requires_grad_(True)

                    if cfg.model.type == 'enn':
                        input_ = (nodes, positions, edges, edge_attr)
                    elif cfg.model.type == 'egnn':
                        n_nodes_arr = [graph.x.size(0)]
                        tensor_n_nodes = torch.tensor(n_nodes_arr)
                        t_n = tensor_n_nodes.repeat_interleave(tensor_n_nodes).unsqueeze(1).to(torch.device(cfg.device))
                        input_ = (nodes, positions, edges, edge_attr, t_n.float())
                    elif cfg.model.type == 'GPSModel':
                        EigVals = graph.EigVals.to(torch.device(cfg.device))
                        EigVecs = graph.EigVecs.to(torch.device(cfg.device))
                        input_ = (nodes, edges, edge_attr, EigVals, EigVecs)

                    elif cfg.model.type=='scgnn':
                        transform = O3Transform(3)
                        graph_Z=Data(x=nodes,edge_index=edges, pos=positions)
                        batchz = torch.arange(0, 1).to(torch.device(cfg.device))
                        n_nodes=nodes.shape[0]
                        graph_Z.batch = batchz.repeat_interleave(n_nodes).long()
                        
                        transform = O3Transform(3)
                        graph_Z = transform(graph_Z)

                        input_ = (
                            graph_Z.x, 
                            graph_Z.pos, 
                            graph_Z.edge_index,
                            graph_Z.node_attr,
                            graph_Z.edge_attr,
                            graph_Z.batch.float()
                            )

                        is_steer=True
                    else:
                        input_ = (nodes, edges, edge_attr)

                    #                 print("~~~~~~~~~Computing Jacobian~~~~~~~~~~~~")

                    node_jacobian = jacobian_graph(model, input_, is_graphgym=is_graphgym, uses_pe=uses_pe,is_steer=is_steer)[0]
                    adj_mat = get_adj_matrix(edges.cpu()).to(torch.device(cfg.device))
                    influence_score, distances = get_influence_score(node_jacobian, adj_mat, positions)

                    dict_ = {
                        "influence_score": influence_score.cpu().detach().numpy(),
                        "xpos": positions.cpu().detach().numpy(),
                        "edges": edges.cpu().detach().numpy(),
                    }
                    entries.append(dict_)

            dump_pkl(entries, file_name=file_name)
            print("Content dumped in pkl file: ", file_name)
