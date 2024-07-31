import os
import sys

sys.path.append('..')

import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchinfo
import torchmetrics.functional as F
import tqdm
from ml_collections import ConfigDict
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from torch_geometric.data import Batch

from chemix import build_chemix, get_mixture_smiles
from chemix.train import LOSS_MAP
from chemix.utils import TORCH_METRIC_FUNCTIONS
from dataloader import DreamLoader
from dataloader.representations.graph_utils import EDGE_DIM, NODE_DIM, from_smiles
from pom.early_stop import EarlyStopping
from pom.gnn.graphnets import GraphNets

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

    # create the pom embedder model
    embedder = GraphNets(node_dim=NODE_DIM, edge_dim=EDGE_DIM, **hp_gnn)
    embedder.load_state_dict(torch.load(f'{fname}/gnn_embedder.pt'), map_location=device
    )
        
    # create the chemix model
    chemix = build_chemix(config=hp_mix.chemix)
    chemix.load_state_dict(torch.load(f'{fname}/chemix.pt'), map_location=device
    )
    torchinfo.summary(chemix)


    ##### SAVE EMBEDDINGS FOR VIS #####
    # train set
    dl = DreamLoader()
    dl.load_benchmark('competition_train_all')
    mixture_labels = dl.features
    dl.featurize('competition_smiles')
    graph_list, indices = get_mixture_smiles(dl.features, from_smiles)
    gr = Batch.from_data_list(graph_list).to(device)
    y = torch.tensor(dl.labels, dtype=torch.float32).to(device)

    single_molecule_indices = np.arange(len(graph_list)).reshape(-1, 1, 1)
    embedder.eval(); chemix.eval()
    with torch.no_grad():
        pom_embed = embedder(gr).squeeze().detach().cpu().numpy()           # pom embedding
        out = embedder.graphs_to_mixtures(gr, single_molecule_indices, device=device)
        single_embed = chemix.embed(out).squeeze().detach().cpu().numpy()   # single mixture embedding
    
    single_mixtures = pd.DataFrame(
        {
            'smiles': [g.smiles for g in graph_list],
            'pom_embedding': [g for g in pom_embed],
            'mixture_embedding': [g for g in single_embed],
        }
    )
    single_mixtures.to_csv('single_molecule_mixtures.csv')

    # Collapse the mixture_labels in the same way indices are collapsed
    indices = indices.transpose(0,2,1).reshape((-1, indices.shape[1]))
    new_data = []
    for row in mixture_labels:
        name = row[0]
        new_data.append([name, row[1]])
        new_data.append([name, row[2]])
    mixture_labels = np.array(new_data, dtype=object)

    _, idx = np.unique(indices, axis=0, return_index=True)
    idx = np.sort(idx)
    unique_indices = np.expand_dims(indices[idx], -1)
    unique_mixtures = mixture_labels[idx]

    # mixture_labels
    with torch.no_grad():
        out = embedder.graphs_to_mixtures(gr, unique_indices, device=device)
        mixture_embeddings = chemix.embed(out).squeeze().detach().cpu().numpy()

    mixture_embedding = pd.DataFrame(
        {
            'dataset': [i for i in unique_mixtures[:, 0]],
            'mixture_label': [i for i in unique_mixtures[:, 1]],
            'mixture_embedding': [g for g in mixture_embeddings],
        }
    )
    mixture_embedding.to_csv('single_mixtures.csv')


    ##### CHECK PERFORMANCE ON THE LEADERBOARD SET #####
    # leaderboard set
    dl_test = DreamLoader()
    dl_test.load_benchmark('competition_leaderboard')
    mixture_labels = dl_test.features
    dl_test.featurize('competition_smiles')
    graph_list, test_indices = get_mixture_smiles(dl_test.features, from_smiles)
    test_gr = Batch.from_data_list(graph_list).to(device)
    y_test = torch.tensor(dl_test.labels, dtype=torch.float32).to(device)

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

    

