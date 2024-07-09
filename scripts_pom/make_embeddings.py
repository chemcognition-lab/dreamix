import sys, os
sys.path.append('..') # required to load dreamloader utilities

from typing import List, Optional

from dataloader.representations import graph_utils
from pom.data import GraphDataset
from pom.gnn.graphnets import GraphNets

import numpy as np
import torch
from torch_geometric.loader import DataLoader as pygdl


def get_embeddings_from_smiles(smi_list: List[str], file_path: str, gnn: Optional[torch.nn.Module] = None):
    # generate a matrix of embeddings
    # Size: [N, embedding_size]
    # enter a model if you want to load a different model, otherwise defaulting
    graphs = [graph_utils.from_smiles(smi, init_globals=True) for smi in smi_list]
    ds = GraphDataset(graphs, np.zeros((len(graphs), 1)))
    loader = pygdl(ds, batch_size=64, shuffle=False)

    if gnn is None:
        gnn = GraphNets.from_json(node_dim=ds.node_dim, edge_dim=ds.edge_dim, json_path=f'{file_path}/hparams.json')
        state_dict = torch.load(f'{file_path}/gnn_embedder.pt', map_location=torch.device('cpu'))
        gnn.load_state_dict(state_dict)

    ## Can add CheMIX model here
    # Pseudo-code:
    # model = WrapperModel(gnn, chemix)
    #
    ## we will need a custom dataset that takes in...
    ### list of mixtures (M)
    ### list of smiles per mixture (L)
    ### (N, M, L)

    embeddings = []
    with torch.no_grad():
        gnn.eval()
        for batch in loader:
            data, _ = batch
            embed = gnn(data)
            embeddings.append(embed)
        embeddings = torch.concat(embeddings, dim=0)

    return embeddings


if __name__ == '__main__':
    embeddings = get_embeddings_from_smiles(['c1ccccc1', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'], 'general_models/graphnets_gs-lf-mayhew/')
    print(embeddings)
