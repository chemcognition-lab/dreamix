import sys, os
sys.path.append('..') # required to load dreamloader utilities

from typing import List, Optional

from dataloader.representations import graph_utils
from pom.data import GraphDataset
from pom.gnn.graphnets import GraphNets
from pom.utils import get_embeddings_from_smiles

import torch
from torch_geometric.loader import DataLoader as pygdl



if __name__ == '__main__':
    get_embeddings_from_smiles(['c1ccccc1', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'])