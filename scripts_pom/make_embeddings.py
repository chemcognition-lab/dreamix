import sys, os
sys.path.append('..') # required to load dreamloader utilities

from typing import List, Optional

from dataloader.representations import graph_utils
from pom.data import GraphDataset
from pom.gnn.graphnets import GraphNets

import torch
from torch_geometric.loader import DataLoader as pygdl

def get_embeddings_from_smiles(smi_list: List[str], model: Optional[torch.nn.Module] = None):
    # generate a matrix of embeddings
    # Size: [N, embedding_size]
    # enter a model if you want to load a different model, otherwise defaulting
    graphs = [graph_utils.from_smiles(smi, init_globals=True) for smi in smi_list]
    dataset = GraphDataset(graphs, torch.zeros(len(graphs), 1))
    loader = pygdl(dataset, batch_size=64, shuffle=False)

    if model is None:
        global_dim = 100
        depth = 2
        model = GraphNets(graphs[0].x.shape[-1], graphs[0].edge_attr.shape[-1], global_dim, depth=depth)
        state_dict = torch.load(f'leffingwell_models/graphnets_d{depth}/gnn_embedder.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    embeddings = []
    with torch.no_grad():
        model.eval()
        for batch in loader:
            data, _ = batch
            embed = model(data)
            embeddings.append(embed)
        embeddings = torch.concat(embeddings, dim=0)

    return embeddings

if __name__ == '__main__':
    get_embeddings_from_smiles(['c1ccccc1', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'])