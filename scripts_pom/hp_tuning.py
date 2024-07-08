import sys, os
sys.path.append('..') # required to load dreamloader utilities

# basic dependencies
from ml_collections import ConfigDict
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

from functools import partial

# dataloader
from dataloader import DataLoader

# pom
import pom.utils as utils
from pom import EndToEndModule
from pom.data import GraphDataset
from pom.early_stop import EarlyStopping
from pom.gnn.graphnets import GraphNets
from pom.gnn.graph_utils import get_graph_degree_histogram, get_pooling_function

# glm
from prediction_head.data import TaskType, get_activation, get_loss_fn
from prediction_head.GLM import GLM

# optuna for tuning
import optuna

import torch
import torchinfo
from torch_geometric.loader import DataLoader as pygdl


def get_split_file(dataset_name, test_size):
    os.makedirs(f'{dataset_name}_models/', exist_ok=True)
    fname = f'{dataset_name}_models/{dataset_name}_{test_size}.npz'
    
    if os.path.isfile(fname):
        split = np.load(fname)
        train_ind, test_ind = split['train_ind'], split['test_ind']
    else:
        dl = DataLoader()
        dl.load_benchmark(dataset_name)
        num_dat = dl.get_dataset_specifications(dataset_name)['n_datapoints']
        labels = dl.labels
        train_ind, _, test_ind, _ = iterative_train_test_split(np.array(range(num_dat)).reshape(-1,1), 
                                                               labels, 
                                                               test_size=test_size)
        np.savez(fname, train_ind=train_ind, test_ind=test_ind)
    return train_ind, test_ind

def objective(trial, train_loader, test_loader, task, task_dim, gr):
    # hparams settings
    hp = ConfigDict({'gnn': ConfigDict(), 'head': ConfigDict(), 'training': ConfigDict()})

    # using graphnets
    hp.gnn.global_dim = trial.suggest_int('global_dim', 100, 200, step=5)     # this is also the embedding space
    hp.gnn.depth = trial.suggest_int('depth', 1, 5)
    hp.gnn.hidden_dim = trial.suggest_int('hidden_dim', 50, 150, step=10)
    hp.gnn.num_layers = trial.suggest_int('num_layers', 1, 3)
    hp.training.num_epochs = 200
    hp.training.lr = 1e-3
    hp.training.batch_size = 64
    hp.training.val_size = 0.2

    seed = 42
    utils.set_seed(seed)
    data_names = ['gs-lf']

    # Load all models and datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_store = {}
    gnn = None
    for dname in data_names:
        gnn = GraphNets(gr.x.shape[-1], gr.edge_attr.shape[-1], hp.gnn.global_dim, depth=hp.gnn.depth).to(device)

        pred = GLM(input_dim=hp.gnn.global_dim, output_dim=task_dim, tasktype=task).to(device)
        model = EndToEndModule(gnn, pred).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.training.lr)

        model_store.update(
            {
                dname: {
                    'model': model,
                    'optimizer': optimizer,
                    'train_loader': train_loader,
                    'test_loader': test_loader,
                    'loss_fn': get_loss_fn(task)(),
                    'metric_fn': utils.get_metric_function(task),
                    'task_type': task
                }
            } 
        )

    # optimization things
    es = EarlyStopping(gnn, patience=15, mode='minimize')       # early stopping only GNN weights
    log = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'val_metric', 'dataset']}
    pbar = tqdm.tqdm(range(hp.training.num_epochs))

    for epoch in pbar:
        avg_train_loss = 0
        avg_test_loss = 0
        for dname, store in model_store.items():
            model = store['model']
            optimizer = store['optimizer']
            train_loader = store['train_loader']
            test_loader = store['test_loader']
            loss_fn = store['loss_fn']
            metric_fn = store['metric_fn']

            # training loop
            log['epoch'].append(epoch)
            log['dataset'].append(dname)
            training_loss = 0
            model.train()
            for batch in train_loader:
                data, y = batch
                data, y = data.to(device), y.to(device)

                optimizer.zero_grad()
                y_hat = model(data)
                loss = loss_fn(y_hat.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
            training_loss /= len(train_loader)
            avg_train_loss += training_loss
            log['train_loss'].append(training_loss)

            # validation loop
            y_pred, y_true = [], []
            with torch.no_grad():
                model.eval()
                for batch in test_loader:
                    data, y = batch
                    data = data.to(device)
                    
                    y_hat = model(data)
                    y_pred.append(y_hat.detach().cpu())
                    y_true.append(y)

                y_pred = torch.concat(y_pred)
                y_true = torch.concat(y_true)
                testing_metric = metric_fn(y_pred.squeeze(), y_true.squeeze()).item()
                testing_loss = loss_fn(y_pred.squeeze(), y_true.squeeze()).item()

            avg_test_loss += testing_loss
            log['val_loss'].append(testing_loss)
            log['val_metric'].append(testing_metric)
            
            # print some statistics
            pbar.set_description(f"Train: {training_loss:.4f} | Test: {testing_loss:.4f} | Test metric: {testing_metric:.4f} | Dataset: {dname}")
            
        # check early stopping based on losses averaged of datasets
        stop = es.check_criteria(avg_test_loss, model.gnn_embedder)        
        if stop:
            print(f'Early stop reached at {es.best_step} with loss {es.best_value}')
            break

    log = pd.DataFrame(log)


    return es.best_value

if __name__ == '__main__':

    os.makedirs('optuna/', exist_ok=True)

    dname = 'gs-lf'
    dl = DataLoader()
    dl.load_benchmark(dname)
    dl.featurize('molecular_graphs', init_globals=True)
    data_specs = dl.get_dataset_specifications(dname)
    task = TaskType(data_specs['task'])
    task_dim = data_specs['task_dim']

    train_ind, test_ind = get_split_file(dname, test_size=0.2)

    features = dl.features
    targets = dl.labels
    dataset = GraphDataset(features, targets)

    # split the data and get dataloaders
    train_set = torch.utils.data.Subset(dataset, train_ind.flatten())
    test_set = torch.utils.data.Subset(dataset, test_ind.flatten())
    train_loader = pygdl(train_set, batch_size=64, shuffle=True)
    test_loader = pygdl(test_set, batch_size=128, shuffle=False)

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, train_loader=train_loader, test_loader=test_loader, task=task, task_dim=task_dim, gr=features[0]), n_trials=100)

    print(study.best_trial.value)
    print()
    print(study.best_params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('optuna/optuna_history.png')

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image('optuna/optuna_coordinates.png')

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image('optuna/optuna_importance.png')

    fig = optuna.visualization.plot_contour(study)
    fig.write_image('optuna/optuna_contour.png')


    
