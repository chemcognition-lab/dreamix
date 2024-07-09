import sys, os
sys.path.append('..') # required to load dreamloader utilities

# basic dependencies
from ml_collections import ConfigDict
import json
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

# dataloader
from dataloader import DataLoader

# pom
import pom.utils as utils
from pom import EndToEndModule
from pom.data import GraphDataset
from pom.early_stop import EarlyStopping
from pom.gnn.graphnets import GraphNets
# from pom.gnn.graph_utils import get_graph_degree_histogram, get_pooling_function

# glm
from prediction_head.data import TaskType, get_loss_fn
from prediction_head.GLM import GLMStructured, TaskSpec

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



if __name__ == '__main__':

    # create folders for logging
    fname = f'general_models/graphnets/'      # create a file name to log all outputs
    os.makedirs(fname, exist_ok=True)

    # hparams settings
    hp = ConfigDict()

    # using graphnets
    hp.global_dim = 128     # this is also the embedding space
    hp.depth = 3
    hp.hidden_dim = 100
    hp.dropout = 0.1
    hp.num_epochs = 200
    hp.lr = 5e-4
    hp.batch_size = 64
    hp.val_size = 0.2
    with open(f'{fname}/hparams.json', 'w') as f:
        f.write(hp.to_json(indent = 4))

    # get dataset names
    seed = 42
    utils.set_seed(seed)
    data_names = DataLoader().get_dataset_names()
    data_names.remove('keller_2016')        # cannot load
    data_names.remove('aromadb_odor')       # cannot load
    data_names.remove('abraham_2012')       # cannot load
    data_names.remove('arctander_1960')     # singleton issue
    data_names.remove('aromadb_descriptor') # singleton issue
    data_names.remove('ifra_2019') # singleton issue
    data_names.remove('flavornet') # singleton issue
    data_names.remove('sharma_2021a') # singleton issue
    data_names.remove('sigma_2014') # singleton issue

    # Load all models and datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    specs = {}
    data_store = {}
    for dname in data_names:
        dl = DataLoader()
        dl.load_benchmark(dname)
        dl.featurize('molecular_graphs', init_globals=True)
        data_specs = dl.get_dataset_specifications(dname)

        # Dataset specifics
        task = TaskType(data_specs['task'])
        task_dim = data_specs['task_dim']
        task_spec = TaskSpec(task_dim, task)
        specs[dname] = task_spec

        train_ind, test_ind = get_split_file(dname, test_size=0.2)

        features = dl.features
        targets = dl.labels
        dataset = GraphDataset(features, targets)

        # split the data and get dataloaders
        train_set = torch.utils.data.Subset(dataset, train_ind.flatten())
        test_set = torch.utils.data.Subset(dataset, test_ind.flatten())
        train_loader = pygdl(train_set, batch_size=hp.batch_size, shuffle=True)
        test_loader = pygdl(test_set, batch_size=128, shuffle=False)      

        # use the same optimizer
        data_store.update(
            {
                dname: {
                    'train_loader': train_loader,
                    'test_loader': test_loader,
                    'loss_fn': get_loss_fn(task)(),
                    'metric_fn': utils.get_metric_function(task),
                    # 'task_spec': task_spec,
                }
            } 
        )

    gr = features[0]
    gnn = GraphNets(gr.x.shape[-1], gr.edge_attr.shape[-1], hp.global_dim, hidden_dim = hp.hidden_dim, depth=hp.depth, dropout = hp.dropout).to(device)
    pred = GLMStructured(input_dim=hp.global_dim, tasks=specs).to(device)
    model = EndToEndModule(gnn, pred).to(device)
    # instantiate
    with torch.no_grad():
        for dname, store in data_store.items():
            for x, y in store['train_loader']:
                model(x, dname)
                break
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)


    # optimization things
    es = EarlyStopping(gnn, patience=20, mode='maximize')       # early stopping only GNN weights
    log = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'val_metric', 'dataset']}
    pbar = tqdm.tqdm(range(hp.num_epochs))

    for epoch in pbar:
        avg_train_loss = 0
        avg_test_loss = 0
        avg_test_metric = 0
        for dname, store in data_store.items():
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
                y_hat = model(data, dname)
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
                    
                    y_hat = model(data, dname)
                    y_pred.append(y_hat.detach().cpu())
                    y_true.append(y)

                y_pred = torch.concat(y_pred)
                y_true = torch.concat(y_true)
                testing_metric = metric_fn(y_pred.squeeze(), y_true.squeeze()).item()
                testing_loss = loss_fn(y_pred.squeeze(), y_true.squeeze()).item()

            avg_test_loss += testing_loss
            avg_test_metric += testing_metric
            log['val_loss'].append(testing_loss)
            log['val_metric'].append(testing_metric)
            
            # print some statistics
            pbar.set_description(f"Train: {training_loss:.4f} | Test: {testing_loss:.4f} | Test metric: {testing_metric:.4f} | Dataset: {dname}")
            
        # check early stopping based on losses averaged of datasets
        avg_train_loss /= len(data_store)
        avg_test_loss /= len(data_store)
        avg_test_metric /= len(data_store)
        log['epoch'].append(epoch)
        log['train_loss'].append(avg_train_loss)
        log['val_loss'].append(avg_test_loss)
        log['val_metric'].append(avg_test_metric)
        log['dataset'].append('average')
        stop = es.check_criteria(avg_test_metric, model.gnn_embedder)        
        if stop:
            print(f'Early stop reached at {es.best_step} with loss {es.best_value}')
            break
    
        # # perform a check between all the models
        # same_weights = True
        # for p1, p2 in zip(model_store['gs-lf']['model'].gnn_embedder.parameters(), model_store['mayhew_2022']['model'].gnn_embedder.parameters()):
        #     if p1.data.ne(p2.data).sum() > 0:
        #         same_weights = False 
        #         break
        # print(f'Weights same: {same_weights} ')

    log = pd.DataFrame(log)
    

    # save model weights
    best_model_dict = es.restore_best()
    model.gnn_embedder.load_state_dict(best_model_dict)      # load the best one trained
    torch.save(model.gnn_embedder.state_dict(), f'{fname}/gnn_embedder.pt')

    # also plot sand save the training loss (for diagnostics)
    log.to_csv(f'{fname}/training.csv', index=False)
    plt_log = log[['epoch', 'val_metric', 'dataset']].melt(id_vars=['epoch', 'dataset'], var_name='set', value_name='metric')
    sns.lineplot(data=plt_log, x='epoch', y='metric', hue='set', style='dataset') 
    plt.savefig(f'{fname}/metric.png', bbox_inches='tight')
    plt.close()

    plt_log = log[['epoch', 'train_loss', 'val_loss', 'dataset']].melt(id_vars=['epoch', 'dataset'], var_name='set', value_name='loss')
    sns.lineplot(data=plt_log, x='epoch', y='loss', hue='set', style='dataset') 
    plt.savefig(f'{fname}/loss.png', bbox_inches='tight')
    plt.close()

    with open(f'{fname}/output.log', 'w') as f:
        f.write(f'Max epochs: {hp.num_epochs}\n')
        f.write(f'Learning rate: {hp.lr}\n')
        f.write(f'Batch size: {hp.batch_size}\n')
        f.write(f'Seed: {seed}\n')
        f.write('------\n')
        f.write(f'Early stop: {es.best_step}\n')
        f.write(f'Stats at early stop epoch: {log.iloc[es.best_step]}\n')
        f.write('------\n')
        f.write('GNN Embedder:\n')
        f.write(f'{torchinfo.summary(gnn)}\n')

