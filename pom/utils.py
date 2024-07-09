from typing import Optional, List, Any

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as pygdl
import torchmetrics.functional as F

import random
import json
import numpy as np

from .gnn import graph_utils, GraphNets
from .data import GraphDataset


def set_seed(seed: int = 42):
    print(f'Seed set to {seed}')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loss_function(task: str):
    # DEPRECATED, use the one in prediction_head.data
    loss_dict = {
        'multilabel': nn.BCEWithLogitsLoss(),
        'binary': nn.BCEWithLogitsLoss(),
        'multiclass': nn.CrossEntropyLoss(),        # also expects logits
    }
    return loss_dict[task]

def multilabel_auroc(pred: torch.Tensor, targ: torch.Tensor):
    # we remove columns that have no meaning in ground truth
    non_zero_ind = torch.sum(targ != 0, dim=0).nonzero().squeeze()
    targ = targ[:, non_zero_ind]
    pred = pred[:, non_zero_ind]
    return F.classification.multilabel_auroc(pred, targ.long(), targ.shape[-1])

def binary_accuracy(pred: torch.Tensor, targ: torch.Tensor):
    pred_class = pred > 0
    targ_class = targ > 0.5
    return ((pred_class.long() - targ_class.long()) == 0).double().mean()

                             
def get_metric_function(task: str):
    loss_dict = {
        'multilabel': lambda pred, targ: F.classification.multilabel_auroc(pred, targ.long(), targ.shape[-1]),
        'multiclass': lambda pred, targ: F.classification.multiclass_auroc(pred, targ.long(), targ.shape[-1]),
        'binary': lambda pred, targ: F.classification.binary_auroc(pred, targ.long()),
        'regression': lambda pred, targ: F.r2_score(pred, targ)
    }
    return loss_dict[task]

def is_jsonable(x: nn.Module):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def jsonify(val: Any):
    if is_jsonable(val): 
        return val
    else:
        return val.__name__
    
    
def get_embeddings_from_smiles(smi_list: List[str], file_path: str, model: Optional[torch.nn.Module] = None):
    # generate a matrix of embeddings
    # Size: [N, embedding_size]
    # enter a model if you want to load a different model, otherwise defaulting
    graphs = [graph_utils.from_smiles(smi, init_globals=True) for smi in smi_list]
    dataset = GraphDataset(graphs, torch.zeros(len(graphs), 1))
    loader = pygdl(dataset, batch_size=64, shuffle=False)

    if model is None:
        model = GraphNets.from_json(f'{file_path}/gnn_hparams.json')
        state_dict = torch.load(f'{file_path}/gnn_embedder.pt', map_location=torch.device('cpu'))
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

        

