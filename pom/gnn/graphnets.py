from typing import Union, List, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer, Linear

# inspired by DIONYSUS (https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/models/modules.py)
# and (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaLayer.html#torch_geometric.nn.models.MetaLayer)

class EdgeMLPModel(nn.Module):
    def __init__(self, edge_dim: int, num_layers: Optional[int] = 1):
        super().__init__()
        self.edge_mlp = get_mlp_module(edge_dim, num_layers)

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dst, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeMLPModel(nn.Module):
    def __init__(self, node_dim: int, num_layers: Optional[int] = 1):
        super().__init__()
        self.node_mlp_1 = get_mlp_module(node_dim, num_layers)
        self.node_mlp_2 = get_mlp_module(node_dim, num_layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalMLPModel(nn.Module):
    def __init__(self, global_dim: int, num_layers: Optional[int] = 1):
        super().__init__()
        self.global_mlp = get_mlp_module(global_dim, num_layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([
            u,
            scatter(x, batch, dim=0, reduce='mean'),
        ], dim=1)
        return self.global_mlp(out)


class GlobalAttnModel(nn.Module):
    def __init__(self, global_dim: int, num_layers: Optional[int] = 1):
        super().__init__()
        self.k_head = get_mlp_module(50, 1)
        self.q_head = get_mlp_module(50, 1)
        self.v_head = get_mlp_module(50, 1)
        self.attention_layer = nn.MultiheadAttention(50*3, num_heads=3, batch_first=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        node_mean = scatter(x, batch, dim=0, reduce='mean')
        # node_min = scatter(x, batch, dim=0, reduce='min')
        # node_max = scatter(x, batch, dim=0, reduce='max')

        K = self.k_head(node_mean)
        Q = self.q_head(node_mean)
        V = self.v_head(node_mean)
        
        import pdb; pdb.set_trace()
        
        out = torch.cat([
            u,
            scatter(x, batch, dim=0, reduce='mean'),
        ], dim=1)
        return self.global_mlp(out)





def get_mlp_module(hidden_dim: int, num_layers: int):
    """
    Helper function to produce MLP with specified hidden dimension and layers
    """
    assert num_layers >= 0, 'Enter an integer larger than or equal to 0.'
    
    layers = nn.ModuleList()
    for _ in range(num_layers):
        layers.append(Linear(-1, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.SELU())
    layers.append(Linear(-1, hidden_dim))
    return nn.Sequential(*layers)


def get_graphnet_mlp_layer(node_dim, edge_dim, global_dim, num_layers = 2):
    """
    Helper function to produced GraphNets layer. 
    """
    node_net = NodeMLPModel(node_dim, num_layers)
    edge_net = EdgeMLPModel(edge_dim, num_layers)
    global_net = GlobalMLPModel(global_dim, num_layers)
    return MetaLayer(edge_net, node_net, global_net)

def get_graphnet_attn_layer(node_dim, edge_dim, global_dim, num_layers = 2):
    """
    Helper function to produced GraphNets layer. 
    """
    node_net = NodeMLPModel(node_dim, num_layers)
    edge_net = EdgeMLPModel(edge_dim, num_layers)
    global_net = GlobalAttnModel(global_dim, num_layers)
    return MetaLayer(edge_net, node_net, global_net)


class GraphNets(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, depth=3):
        super(GraphNets, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.layers = nn.ModuleList(
            [
                get_graphnet_mlp_layer(node_dim, edge_dim, global_dim) for _ in range(depth)
            ]
        )

    def forward(self, data):
        x, edge_index, edge_attr, num_nodes, u, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.num_nodes,
            data.u,
            data.batch,
        )

        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)
        
        return u


