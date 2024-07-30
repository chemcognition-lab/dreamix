import sys, os
sys.path.append('..')

import json
import tqdm
from ml_collections import ConfigDict

import torch
import torch.nn as nn
from torch_geometric.data import Batch
import torchmetrics.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau

from dataloader import DreamLoader
from dataloader.representations.graph_utils import from_smiles, NODE_DIM, EDGE_DIM
from pom.gnn.graphnets import GraphNets
from pom.early_stop import EarlyStopping
from chemix import get_mixture_smiles, build_chemix
from chemix.train import LOSS_MAP
from chemix.utils import TORCH_METRIC_FUNCTIONS

import torchinfo

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--trial", action="store", type=int, default=1, help="Trial number.")
parser.add_argument("--no-verbose", action="store_true", default=False)
FLAGS = parser.parse_args()

if __name__ == '__main__':
    fname = f'results/chemix_ensemble/top{FLAGS.trial}/'      # this is the checkpoint folder

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on: {device}')

    # load hyperparameters from the file
    hp_gnn = ConfigDict(json.load(open(f'{fname}/hparams_graphnets.json', 'r')))
    hp_mix = ConfigDict(json.load(open(f'{fname}/hparams_chemix.json', 'r')))

    # leaderboard set
    dl_test = DreamLoader()
    dl_test.load_benchmark('competition_leaderboard')
    dl_test.featurize('competition_smiles')
    graph_list, test_indices = get_mixture_smiles(dl_test.features, from_smiles)
    test_gr = Batch.from_data_list(graph_list)
    y_test = torch.tensor(dl_test.labels, dtype=torch.float32).to(device)

    # create the pom embedder model
    embedder = GraphNets(node_dim=NODE_DIM, edge_dim=EDGE_DIM, **hp_gnn)
    embedder.load_state_dict(torch.load(f'{fname}/gnn_embedder.pt'))
    embedder = embedder.to(device)
    
    # create the chemix model
    chemix = build_chemix(config=hp_mix.chemix)
    chemix.load_state_dict(torch.load(f'{fname}/chemix.pt'))
    chemix = chemix.to(device=device)
    torchinfo.summary(chemix)
    

    ### CHECK PERFORMANCE ON THE LEADERBOARD SET
    embedder.eval(); chemix.eval()
    with torch.no_grad():
        out = embedder.graphs_to_mixtures(test_gr, test_indices, device=device)
        y_pred = chemix(out)
    
    # calculate a bunch of metrics on the results to compare
    leaderboard_metrics = {}
    for name, func in TORCH_METRIC_FUNCTIONS.items():
        leaderboard_metrics[name] = func(y_pred.flatten(), y_test.flatten()).detach().cpu().numpy()
    leaderboard_metrics = pd.DataFrame(leaderboard_metrics, index=['metrics']).transpose()

    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_test = y_test.detach().cpu().numpy().flatten()
    leaderboard_predictions = pd.DataFrame({
        'Predicted_Experimental_Values': y_pred, 
        'Ground_Truth': y_test,
        'MAE': np.abs(y_pred - y_test),
    }, index=range(len(y_pred)))

    # plot the predictions
    ax = sns.scatterplot(data=leaderboard_predictions, x='Ground_Truth', y='Predicted_Experimental_Values')
    ax.plot([0,1], [0,1], 'r--')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.annotate(''.join(f'{k}: {v['metrics']:.4f}\n' for k, v in leaderboard_metrics.iterrows()).strip(),
            xy=(0.05,0.7), xycoords='axes fraction',
            # textcoords='offset points',
            size=12,
            bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))
    plt.savefig(f'{fname}/reload_predictions.png', bbox_inches='tight')
    plt.close()

    

