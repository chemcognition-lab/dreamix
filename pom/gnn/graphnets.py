from typing import Union, List, Optional

import json

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.nn import MetaLayer, Linear, GATConv, GAT
from torch_geometric.nn.aggr import MultiAggregation

# inspired by DIONYSUS (https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/models/modules.py)
# and (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaLayer.html#torch_geometric.nn.models.MetaLayer)
    
class EdgeFiLMModel(nn.Module):
    def __init__(self, 
                 edge_dim: int, 
                 hidden_dim: Optional[int] = 50, 
                 num_layers: Optional[int] = 1, 
                 dropout: Optional[float] = 0.,
                ):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # FiLM
        self.gamma = get_mlp(hidden_dim, edge_dim, num_layers, dropout=dropout)
        self.gamma_act = nn.Sigmoid()       # sigmoidal gating Dauphin et al.
        self.beta = get_mlp(hidden_dim, edge_dim, num_layers, dropout=dropout)

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        cond = torch.cat([src, dst, u[batch]], 1)
        gamma = self.gamma_act(self.gamma(cond))
        beta = self.beta(cond)

        return gamma * edge_attr + beta
    

class NodeAttnModel(nn.Module):
    def __init__(self, 
                 node_dim: int, 
                 hidden_dim: Optional[int] = 50, 
                 num_heads: Optional[int] = 5,
                 dropout: Optional[int] = 0., 
                 num_layers: Optional[int] = 1,
                ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers

        # self attention layer
        self.self_attn = GAT(node_dim, node_dim, num_layers=num_layers, dropout=dropout, v2=True, heads=num_heads)
        self.output_mlp = get_mlp(hidden_dim, node_dim, num_layers=2)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u: torch.Tensor, batch: torch.Tensor):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        attn = self.self_attn(x, edge_index, edge_attr)
        out = self.norm1(x + self.dropout_layer(attn))
        out = self.norm2(out + self.dropout_layer(self.output_mlp(out)))
        return out
    

class GlobalPNAModel(nn.Module):
    def __init__(self, 
                 global_dim: int, 
                 hidden_dim: Optional[int] = 50, 
                 num_layers: Optional[int] = 2,
                 dropout: Optional[float] = 0.,
                ):
        super().__init__()
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.pool = MultiAggregation(["mean", "std", "max", "min"])
        self.global_mlp = get_mlp(hidden_dim, global_dim, num_layers, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u: torch.Tensor, batch: torch.Tensor):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        aggr = self.pool(x, batch)
        out = torch.cat([u, aggr], dim=1)
        return self.global_mlp(out)


##### Helper functions #####

def get_mlp(hidden_dim: int, output_dim: int, num_layers: int, dropout: Optional[float] = 0.):
    """
    Helper function to produce MLP with specified hidden dimension and layers
    """
    assert num_layers > 0, 'Enter an integer larger than 0.'
    
    layers = nn.ModuleList()
    for _ in range(num_layers-1):
        layers.append(Linear(-1, hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.SELU())
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(Linear(-1, output_dim))
    return nn.Sequential(*layers)


def get_graphnet_layer(
        node_dim: int, 
        edge_dim: int, 
        global_dim: int,
        hidden_dim: Optional[int] = 50,
        dropout: Optional[float] = 0.,
    ):
    """
    Helper function to produce GraphNets layer. 
    """
    node_net = NodeAttnModel(node_dim, hidden_dim=hidden_dim, dropout=dropout)
    edge_net = EdgeFiLMModel(edge_dim, hidden_dim=hidden_dim, dropout=dropout)
    global_net = GlobalPNAModel(global_dim, hidden_dim=hidden_dim, dropout=dropout)
    return MetaLayer(edge_net, node_net, global_net)


class GraphNets(nn.Module):
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int, 
                 global_dim: int, 
                 hidden_dim: Optional[int] = 50,
                 depth: Optional[int] =  3,
                 dropout: Optional[float] = 0.,
                 **kwargs
                ):
        super(GraphNets, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.depth = depth
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                get_graphnet_layer(
                    node_dim, 
                    edge_dim, 
                    global_dim, 
                    hidden_dim=hidden_dim, 
                    dropout=dropout,
                ) for _ in range(depth)
            ]
        )

    def forward(self, data: pyg.data.Data):
        x, edge_index, edge_attr, u, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.u,
            data.batch,
        )

        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)
        
        return u
    
    @classmethod
    def from_json(cls, json_path: str):
        params = json.load(open(json_path, 'r'))
        return cls(**params)


