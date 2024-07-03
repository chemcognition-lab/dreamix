import torch
import torch.nn as nn
import torchmetrics.functional as F

import random
import json
import numpy as np


def set_seed(seed: int = 42):
    print(f'Seed set to {seed}')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loss_function(task: str):
    loss_dict = {
        'multilabel': nn.BCEWithLogitsLoss(),
        'binary': nn.BCEWithLogitsLoss(),
        'multiclass': nn.CrossEntropyLoss(),        # also expects logits
    }
    return loss_dict[task]

def multilabel_auroc(pred, targ):
    # we remove columns that have no meaning in ground truth
    non_zero_ind = torch.sum(targ != 0, dim=0).nonzero().squeeze()
    targ = targ[:, non_zero_ind]
    pred = pred[:, non_zero_ind]
    return F.classification.multilabel_auroc(pred, targ.long(), targ.shape[-1])

def binary_accuracy(pred, targ):
    pred_class = pred > 0
    targ_class = targ > 0.5
    return ((pred_class.long() - targ_class.long()) == 0).double().mean()

                             
def get_metric_function(task: str):
    loss_dict = {
        'multilabel': lambda pred, targ: F.classification.multilabel_auroc(pred, targ.long(), targ.shape[-1]),
        'binary': lambda pred, targ: F.classification.binary_auroc(pred, targ.long()),
        'multiclass': lambda pred, targ: F.classification.multiclass_auroc(pred, targ.long(), targ.shape[-1]),
    }
    return loss_dict[task]

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def jsonify(val):
    if is_jsonable(val): 
        return val
    else:
        return val.__name__

        

