"""


Example usage:
```
python scripts_chemix/load_chemix.py
```

"""

import sys

sys.path.append("..")
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics.functional as F
from ml_collections import ConfigDict
from torch_geometric.data import Batch

from chemix import Chemix, MixtureNet, Regressor
from chemix.data import get_mixture_smiles
from chemix.model import AttentionAggregation
from dataloader import DreamLoader
from dataloader.representations.graph_utils import EDGE_DIM, NODE_DIM, from_smiles
from pom.gnn.graphnets import GraphNets

EXPECTED_VALUE = 0.77
if __name__ == "__main__":
    fname = "chemix_end2end/"  # this is the checkpoint folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # using hyperparameters
    hp_gnn = ConfigDict(json.load(open(f"{fname}/hparams_graphnets.json", "r")))
    hp_mix = ConfigDict(json.load(open(f"{fname}/hparams_chemix.json", "r")))

    # training set
    dl = DreamLoader()
    dl.load_benchmark("competition_train")
    dl.featurize("competition_smiles")
    graph_list, train_indices = get_mixture_smiles(dl.features, from_smiles)
    train_gr = Batch.from_data_list(graph_list)
    y_train = torch.tensor(dl.labels, dtype=torch.float32).to(device)

    # testing set
    dl_test = DreamLoader()
    dl_test.load_benchmark("competition_leaderboard")
    dl_test.featurize("competition_smiles")
    graph_list, test_indices = get_mixture_smiles(dl_test.features, from_smiles)
    test_gr = Batch.from_data_list(graph_list)
    y_test = torch.tensor(dl_test.labels, dtype=torch.float32).to(device)

    # create the pom embedder model
    embedder = GraphNets(node_dim=NODE_DIM, edge_dim=EDGE_DIM, **hp_gnn)
    embedder.load_state_dict(
        torch.load(f"{fname}/gnn_embedder.pt", map_location=device)
    )
    # create the chemix model
    mixture_net = MixtureNet(
        num_layers=hp_mix.num_layers,
        embed_dim=hp_mix.embed_dim,
        num_heads=1,
        mol_aggregation=AttentionAggregation(hp_mix.embed_dim),
        dropout_rate=hp_mix.dropout,
    )

    regressor = Regressor(
        hidden_dim=2 * hp_mix.embed_dim,  # multiply by 4 for pna
        output_dim=1,
        num_layers=hp_mix.num_layers,
        dropout_rate=hp_mix.dropout,
    )

    chemix = Chemix(
        input_net=None,
        regressor=regressor,
        mixture_net=mixture_net,
        unk_token=hp_mix.unk_token,
    ).to(device)

    # torchinfo.summary(chemix)
    chemix.load_state_dict(torch.load(f"{fname}/chemix.pt", map_location=device))

    # This is a reload run, just to sanity check
    embedder.eval()
    chemix.eval()
    with torch.no_grad():
        out = embedder.graphs_to_mixtures(test_gr, test_indices, device=device)
        y_pred = chemix(out)

    rho = F.pearson_corrcoef(y_pred.flatten(), y_test.flatten())
    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_test = y_test.detach().cpu().numpy().flatten()
    leaderboard_predictions = pd.DataFrame(
        {
            "Predicted_Experimental_Values": y_pred,
            "Ground_Truth": y_test,
            "MAE": np.abs(y_pred - y_test),
        },
        index=range(len(y_pred)),
    )
    # leaderboard_predictions.to_csv(f'{fname}/predictions.csv')
    if rho < EXPECTED_VALUE:
        warnings.warn(f"Expected Pearson R of {EXPECTED_VALUE}, got {rho:.5f}")
    else:
        print(f"PEARSON R = {rho:.5f}")

    # plot the predictions
    sns.scatterplot(
        data=leaderboard_predictions,
        x="Ground_Truth",
        y="Predicted_Experimental_Values",
    )
    plt.plot([0, 1], [0, 1], "r--", label=f"Pearson: {rho:.5f}")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(f"{fname}/sanity_check.png", bbox_inches="tight")
    plt.close()
