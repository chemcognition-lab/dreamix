import sys
sys.path.append('..')  # required to load dreamloader utilities

import numpy as np
import torch
import pickle
import os
from dataloader import DreamLoader
from scripts_pom.make_embeddings import get_embeddings_from_smiles


if __name__ == "__main__":
    DATA_PATH = "/Users/ellarajaonson/Documents/dream/sean_ds_def/"
    TRAIN_DATAPATH = os.path.join(DATA_PATH, "dataset_pickle_tmp")
    LEADERBOARD_DATAPATH = os.path.join(DATA_PATH, "dataset_pickle_leaderboard")
    UNK_TOKEN = -999

    # Training set
    data = DreamLoader()
    data.load_benchmark("competition_train")
    data.featurize("competition_smiles")

    X = data.features

    print(X[0, 0])

    max_pad_len_X1 = max(len(arr) for arr in X[:, 0])
    max_pad_len_X2 = max(len(arr) for arr in X[:, 1])

    max_pad_len = max(max_pad_len_X1, max_pad_len_X2)

    X_1 = torch.Tensor([np.pad(get_embeddings_from_smiles(mix), ((0, max_pad_len - get_embeddings_from_smiles(mix).shape[0]), (0, 0)), mode='constant', constant_values=UNK_TOKEN) for mix in X[:, 0]]).to(torch.float32)
    X_2 = torch.Tensor([np.pad(get_embeddings_from_smiles(mix), ((0, max_pad_len - get_embeddings_from_smiles(mix).shape[0]), (0, 0)), mode='constant', constant_values=UNK_TOKEN) for mix in X[:, 1]]).to(torch.float32)

    y = torch.Tensor(data.labels).flatten().to(torch.float32)

    with open(os.path.join(TRAIN_DATAPATH, "x1.pkl"), "wb") as f:
        pickle.dump(X_1, f)
    with open(os.path.join(TRAIN_DATAPATH, "x2.pkl"), "wb") as f:
        pickle.dump(X_2, f)
    with open(os.path.join(TRAIN_DATAPATH, "y.pkl"), "wb") as f:
        pickle.dump(y, f)

    # Leaderboard set
    data.load_benchmark("competition_leaderboard")
    data.featurize("competition_smiles")

    X = data.features
    max_pad_len_X1 = max(len(arr) for arr in X[:, 0])
    max_pad_len_X2 = max(len(arr) for arr in X[:, 1])

    max_pad_len = max(max_pad_len_X1, max_pad_len_X2)

    X_1_leaderboard = torch.Tensor([np.pad(get_embeddings_from_smiles(mix), ((0, max_pad_len - get_embeddings_from_smiles(mix).shape[0]), (0, 0)), mode='constant', constant_values=UNK_TOKEN) for mix in X[:, 0]]).to(torch.float32)
    X_2_leaderboard = torch.Tensor([np.pad(get_embeddings_from_smiles(mix), ((0, max_pad_len - get_embeddings_from_smiles(mix).shape[0]), (0, 0)), mode='constant', constant_values=UNK_TOKEN) for mix in X[:, 1]]).to(torch.float32)

    with open(os.path.join(LEADERBOARD_DATAPATH, "x1.pkl"), "wb") as f:
        pickle.dump(X_1_leaderboard, f)
    with open(os.path.join(LEADERBOARD_DATAPATH, "x2.pkl"), "wb") as f:
        pickle.dump(X_2_leaderboard, f)
