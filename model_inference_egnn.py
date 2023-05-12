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
import pickle
# model = torch.load("/home/lcur1702/lrgb_madhura/lrgb-uva-dl2-11/results/vocsuperpixels-EGNN/0/ckpt/999.ckpt",
#         map_location=torch.device('cpu'))
# print(model)
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def compute_jacobian(inputs, output):
	"""
	:param inputs: Batch X Size (e.g. Depth X Width X Height)
	:param output: Batch X Classes
	:return: jacobian: Batch X Classes X Size
	"""
	assert inputs.requires_grad

	num_classes = output.size()[1]

	jacobian = torch.zeros(num_classes, *inputs.size())
	grad_output = torch.zeros(*output.size())
	if inputs.is_cuda:
		grad_output = grad_output.cuda()
		jacobian = jacobian.cuda()

	for i in range(num_classes):
		zero_gradients(inputs)
		grad_output.zero_()
		grad_output[:, i] = 1
		output.backward(grad_output, retain_graph=True)
		jacobian[i] = inputs.grad.data

	return torch.transpose(jacobian, dim0=0, dim1=1)


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
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = custom_egnn.EGNN(in_node_nf=12, in_edge_nf=0, hidden_nf=128, n_layers=7, coords_weight=1.0,device=cfg.device)

        
        if cfg.train.finetune: 
            model = init_model_from_pretrained(model, cfg.train.finetune,
                                               cfg.train.freeze_pretrained)
            # print(model)
              
            for batch in loaders[0]:
                batch.split = "test"
                graph = batch[0]
                # for i in batch:
                #     print("graph: ", graph[0])
                
                nodes = graph.x[:,:12].to(torch.device(cfg.device))
                positions = graph.x[:,12:].to(torch.device(cfg.device))

                edges = graph.edge_index.to(torch.device(cfg.device))
                edge_attr = graph.edge_attr.to(torch.device(cfg.device))
                print(nodes.size(), edges.size(), edge_attr.size())
    
                true = graph.y
                nodes.requires_grad_(True)
                positions.requires_grad_(True)
                edges = edges.float()
                edges.requires_grad_(False)
                edge_attr.requires_grad_(True)
            
                # pred = model(h0=nodes, x=positions, edges=edges, edge_attr=edge_attr)
                # print("predictions done")  
                    
                input_ = (nodes,positions,edges, edge_attr)
                # model = model.cpu()
                # nodes, positions, edges, edge_attr = nodes.cpu(), positions.cpu(), edges.cpu(), edge_attr.cpu()
                print("jacobian strted")
                jacobian = torch.autograd.functional.jacobian(model, input_)
                # print("deriv done", jacobian.size())
                # node_jacobian = jacobian[0]
                # node_jacobian = node_jacobian.mean(dim=(1,3))
                file = open('jac.pkl', 'wb')
                pickle.dump(jacobian, file)
                file.close()

                file2 = open('pos.pkl', 'wb')
                pickle.dump(positions, file2)
                file2.close()

                print(jacobian[0].size())
                print(jacobian[1].size())
                print(jacobian[2].size())
                print(jacobian[3].size())
                print(jacobian, file=open("trial.txt", "a"))
                break
            
               
       