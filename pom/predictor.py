from typing import List, Iterable, Union, Optional, Callable

import numpy as np
import rdkit.Chem.AllChem as Chem
import json

from . import utils

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import Linear

class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 100, num_layers: int = 2, output_dim: int = 1, dropout_rate: float = 0.1):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential(Linear(-1, hidden_dim), nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        output = self.layers(x)
        return output



class EndToEndModule(nn.Module):
    def __init__(self, gnn_embedder: nn.Module, nn_predictor: nn.Module):
        super(EndToEndModule, self).__init__()
        self.gnn_embedder = gnn_embedder
        self.nn_predictor = nn_predictor
    
    def forward(self, data):
        embedding = self.gnn_embedder(data)
        output = self.nn_predictor(embedding)
        return output
    
    def save_hyperparameters(self, fname):
        hparams = {'gnn': {}, 'head': {}}
        for attr, val in vars(self.gnn_embedder).items():
            if not attr.startswith('_'): 
                hparams['gnn'][attr] = utils.jsonify(val)
        for attr, val in vars(self.nn_predictor).items():
            if not attr.startswith('_'): 
                hparams['head'][attr] = utils.jsonify(val)
                
        if not fname.endswith('.json'):
            fname += 'pom_hparams.json'


        json.dump(hparams, open(fname, 'w'), indent=4)
            

