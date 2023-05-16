import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch import nn

from .gcl import E_GCL, unsorted_segment_sum

class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        # del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) 
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord = coord + agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index # edge index shape = 2 X E
        radial, coord_diff = self.coord2radial(edge_index, coord) # E X 1 , E X 2     
        edge_feat = self.edge_model(h[row.long()], h[col.long()], radial, edge_attr)
        
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

class EGNN(torch.nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=torch.nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))
        
        # For concatentation 
        # self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf*n_layers, self.hidden_nf), 
        #                               act_fn,
        #                               nn.Linear(self.hidden_nf, self.hidden_nf))
        
        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 21))
        
        # For concatenation
        # self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf*n_layers, self.hidden_nf),
        #                                act_fn,
        #                                nn.Linear(self.hidden_nf, 21))
        
        self.to(self.device)

  
      
    def forward(self, h0, x, edges, edge_attr):
        h = self.embedding(h0)
        layer_outputs = []
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_attr=h0)
                layer_outputs.append(h)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_attr=None)
        
        # for addition
        # if self.node_attr:  
        #   h = torch.cat(layer_outputs, dim=1)

        h = self.node_dec(h)

        # for maximum
        # h1 = torch.stack(layer_outputs, dim=0)
        # h = torch.max(h1, dim=0)[0]

        pred = self.graph_dec(h)
        return pred


class E_GCL_mask2(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        # del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, x_weights):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) 
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord_update = torch.div(agg, x_weights)
        coord = coord + coord_update
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, x_weights=None):
        row, col = edge_index # edge index shape = 2 X E
        radial, coord_diff = self.coord2radial(edge_index, coord) # E X 1 , E X 2     
        edge_feat = self.edge_model(h[row.long()], h[col.long()], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, x_weights)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

class EGNN2(torch.nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=torch.nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN2, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask2(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))
        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 21))

# #         For concatenation
#         self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf*n_layers, self.hidden_nf),
#                                        act_fn,
#                                        nn.Linear(self.hidden_nf, 21))
        self.to(self.device)

  
      
    def forward(self, h0, x, edges, edge_attr, x_weights):
        h = self.embedding(h0)
        layer_outputs = []
        for i in range(0, self.n_layers):
            x = x - x.min(0, keepdim=True)[0]
            x = x / x.max(0, keepdim=True)[0]
            if self.node_attr: 
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_attr=h0, x_weights=x_weights)
                layer_outputs.append(h)
            else:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_attr=None, x_weights=x_weights)

        h = self.node_dec(h)

#         # for addition
#         if self.node_attr:  
#             h = torch.cat(layer_outputs, dim=1)
        
#         h1 = torch.stack(layer_outputs, dim=0)
#         h = torch.max(h1, dim=0)[0]

        pred = self.graph_dec(h)
        return pred
    

register_network('custom_egnn', EGNN)
register_network('custom_egnn2', EGNN2)