import sys, os
sys.path.append('..') # required to load dreamloader utilities

from ml_collections import ConfigDict

import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np
import seaborn as sns
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

from dataloader import DataLoader
import pom.utils as utils
from pom import EndToEndModule, MLP
from pom.data import GraphDataset
from pom.early_stop import EarlyStopping
from pom.gnn.chemprop import RevIndexedData, ChemProp
from pom.gnn.graphnets import GraphNets
from pom.gnn.graph_utils import get_graph_degree_histogram, get_pooling_function, get_model_parameters_count

import torch
import torchinfo

from torch_geometric.loader import DataLoader as pygdl


def plot_islands(bg_points,
                 bg_name,
                 *fg_specs,
                 alpha = 0.4,
                 z_limit = 15,
                 colors=None):
    """Plot islands in a background sea of points.

    This command executes matplotlib.pyplot commands as side effects.
    Use plt.figure() to control where these outputs get generated.

    Args:
    bg_points: Array of shape (n_points, 2) indicating x,y coordinates
        of any background points.
    bg_name: Name of background points
    *fg_specs: Repeated island specifications. Allowable dict keys:
        data - array of shape (n_points, 2)
        label - str, name of the island
        color - RGB, color of the island
        scatter - bool, whether to add the point scatters
        scatter_size - int, radius of scatter points in pixel
        filled - bool, whether to fill the island
        levels_to_plot - List[float], percentiles of the KDE to plot.
    alpha: Transparency of island color fill.
    z_limit: Plotting boundaries of the plot.
    """
    default_fg_colors = colors or sns.color_palette('viridis', len(fg_specs))

    sns.scatterplot(x=bg_points[:, 0], y=bg_points[:, 1],
              s=3, color='0.60', label=bg_name)
    for fg_color, fg_spec in zip(default_fg_colors, fg_specs):
        fg_color = fg_spec.get('color', fg_color)
        fg_scatter = fg_spec.get('scatter', False)
        fg_scatter_size = fg_spec.get('scatter_size', 5)
        fg_filled = fg_spec.get('filled', False)
        fg_level_to_plot = fg_spec.get('level_to_plot', 0.25)
        x, y = fg_spec['data'][:, 0], fg_spec['data'][:, 1]
        label = fg_spec['label']
        if fg_scatter:
            sns.scatterplot(x, y, s=fg_scatter_size, color=fg_color)
        if fg_filled:
            sns.kdeplot(x=x, y=y, color=fg_color, fill=True,
                      thresh=fg_level_to_plot, alpha=alpha, levels=2, bw_method=0.3)
        else:
            sns.kdeplot(x=x, y=y, color=fg_color, fill=False,
                      thresh=fg_level_to_plot, levels=2, bw_method=0.3)
        # Generate the legend entry
        if fg_filled:
            plt.scatter([], [], marker='s', c=fg_color, label=label)
        else:
            plt.plot([], [], c=fg_color, linewidth=3, label=label)
    plt.gca().set_aspect('equal', adjustable='box')
    

def plot_odor_islands(pca_space, pc_data, z_limit=15):
    plt.figure(figsize=(12, 8))
    smells = ['jasmine', 'beefy', 'fermented']
    color_palette = sns.color_palette('rocket', len(smells))
    fg_specs = []
    for i, subgroup in enumerate(smells):
        island_embeddings = pca_space[pc_data[subgroup]]
        fg_specs.append({'data': island_embeddings,
                        'filled': False,
                        'scatter': False,
                        'level_to_plot': 0.2,
                        'label': subgroup.capitalize()})
    plot_islands(pca_space, None, *fg_specs, colors=color_palette, z_limit=z_limit)
    fg_specs = []
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))



if __name__ == '__main__':
    # hparams settings
    hp = ConfigDict({
        'gnn': ConfigDict(),
        'head': ConfigDict(),
        'training': ConfigDict()
    })

    # using chemprop
    # hp.gnn.hidden_dim = 100
    # hp.gnn.depth = 3
    hp.gnn.pooling = 'pna'

    # using graphnets
    hp.gnn.hidden_dim = 100
    hp.gnn.global_dim = 100
    hp.gnn.depth = 1

    hp.head.hidden_dim = 100
    hp.head.dropout_rate = 0.1

    hp.training.num_epochs = 100
    hp.training.lr = 1e-3
    hp.training.batch_size = 64
    hp.training.task = 'multilabel'
    hp.training.val_size = 0.2

    
    # create folders for logging
    fname = f'trained_models/graphnets_d{hp.gnn.depth}_attention/'      # create a file name to log all outputs
    os.makedirs(fname, exist_ok=True)
    
    # set seed for reproducibility
    seed = 42
    utils.set_seed(seed)

    # load the data
    dl = DataLoader()
    dl.load_benchmark('leffingwell')
    dl.featurize('molecular_graphs', init_globals=True)
    gr = dl.features[0]
    print(f'Example of graph: {gr}')

    # setup dataset
    # features = [RevIndexedData(g) for g in dl.features] # include additional mp based on chemprop architecture
    features = dl.features
    targets = dl.labels
    train_ind, _, test_ind, _ = iterative_train_test_split(np.array(range(len(features))).reshape(-1,1), targets, test_size=hp.training.val_size)

    dataset = GraphDataset(features, targets)
    train_set = torch.utils.data.Subset(dataset, train_ind.flatten())
    test_set = torch.utils.data.Subset(dataset, test_ind.flatten())
    train_loader = pygdl(train_set, batch_size=hp.training.batch_size, shuffle=True)
    test_loader = pygdl(test_set, batch_size=128, shuffle=False)
    
    # get functions based on task
    loss_fn = utils.get_loss_function(hp.training.task)
    metric_fn = utils.get_metric_function(hp.training.task)
    pooling_fn = get_pooling_function(hp.gnn.pooling, {'deg': get_graph_degree_histogram(train_set)})

    # make model
    # transfer to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # gnn = ChemProp(hp.gnn.hidden_dim, node_dim=features[0].x.shape[-1], edge_dim=features[0].edge_attr.shape[-1], 
    #                pooling_fn=pooling_fn, depth=hp.gnn.depth).to(device)
    gnn = GraphNets(gr.x.shape[-1], gr.edge_attr.shape[-1], hp.gnn.global_dim, depth=hp.gnn.depth).to(device)
    mlp = MLP(hidden_dim=hp.head.hidden_dim, output_dim=targets[0].shape[-1], dropout_rate=hp.head.dropout_rate).to(device)
    model = EndToEndModule(gnn, mlp).to(device)


    # optimization things
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.training.lr)
    es = EarlyStopping(model, patience=20, mode='minimize')

    # train with early stopping
    log = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'val_metric']}
    pbar = tqdm.tqdm(range(hp.training.num_epochs))
    for epoch in pbar:
        training_loss = 0
        model.train()

        # training loop
        log['epoch'].append(epoch)
        for batch in train_loader:
            data, y = batch
            data, y = data.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(data)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        training_loss /= len(train_loader)
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
            testing_metric = metric_fn(y_pred, y_true).item()
            testing_loss = loss_fn(y_pred, y_true).item()
        log['val_loss'].append(testing_loss)
        log['val_metric'].append(testing_metric)
        
        # print some statistics
        pbar.set_description(f"Train loss: {training_loss:.4f} | Test loss: {testing_loss:.4f} | Test metric: {testing_metric:.4f}")
        
        # check early stopping
        stop = es.check_criteria(testing_loss, model)        
        if stop:
            print(f'Early stop reached at {es.best_step} with loss {es.best_value}')
            break

    log = pd.DataFrame(log)

    # save all outputs
    model.save_hyperparameters(f'{fname}/')
    best_model_dict = es.restore_best()
    torch.save(best_model_dict, f'{fname}/model_state.pt')
    log.to_csv(f'{fname}/training.csv', index=False)
    # also plot sand save the training loss (for diagnostics)
    plt_log = log[['epoch', 'train_loss', 'val_loss']].melt(id_vars='epoch', var_name='set', value_name='loss')
    sns.lineplot(data=plt_log, x='epoch', y='loss', hue='set') 
    plt.savefig(f'{fname}/loss.png', bbox_inches='tight')
    plt.close()


    with open(f'{fname}/output.log', 'w') as f:
        f.write(f'Max epochs: {hp.training.num_epochs}\n')
        f.write(f'Learning rate: {hp.training.lr}\n')
        f.write(f'Batch size: {hp.training.batch_size}\n')
        f.write(f'Seed: {seed}\n')
        f.write('------\n')
        f.write(f'Early stop: {es.best_step}\n')
        f.write(f'Stats at early stop epoch: {log.iloc[es.best_step]}\n')
        f.write('------\n')
        f.write('GNN Embedder:\n')
        f.write(f'{torchinfo.summary(gnn)}\n')
    

    # print model parameters
    print(f'GNN embedder: {get_model_parameters_count(gnn)}')
    print(f'Predictor: {get_model_parameters_count(mlp)}')
    print(f'Full num params: {get_model_parameters_count(model)}')


    ###################### DRAWING POM #####################
    # draw the POM
    model.load_state_dict(best_model_dict)      # load the best one trained
    model = model.to(torch.device('cpu'))       # doing it on the 
    all_loader = pygdl(dataset, batch_size=128, shuffle=False)
    keys = ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'animal', 'anisic', 'apple', 'apricot', 
            'aromatic', 'balsamic', 'banana', 'beefy', 'berry', 'black currant', 'brandy', 'bread', 
            'brothy', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'catty', 'chamomile', 
            'cheesy', 'cherry', 'chicken', 'chocolate', 'cinnamon', 'citrus', 'cocoa', 'coconut', 
            'coffee', 'cognac', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 
            'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruity', 'garlic', 'gasoline', 'grape', 
            'grapefruit', 'grassy', 'green', 'hay', 'hazelnut', 'herbal', 'honey', 'horseradish', 'jasmine', 
            'ketonic', 'leafy', 'leathery', 'lemon', 'malty', 'meaty', 'medicinal', 'melon', 'metallic', 
            'milky', 'mint', 'mushroom', 'musk', 'musty', 'nutty', 'odorless', 'oily', 'onion', 'orange', 
            'orris', 'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 
            'pungent', 'radish', 'ripe', 'roasted', 'rose', 'rum', 'savory', 'sharp', 'smoky', 'solvent', 
            'sour', 'spicy', 'strawberry', 'sulfurous', 'sweet', 'tea', 'tobacco', 'tomato', 'tropical', 
            'vanilla', 'vegetable', 'violet', 'warm', 'waxy', 'winey', 'woody']
    results = []
    with torch.no_grad():
        model.eval()
        for batch in all_loader:
            res = {}
            data, y = batch
            embed = model.gnn_embedder(data)
            res.update({'embedding': embed.detach().numpy().tolist()})
            res.update({k: v for k,  v in zip(keys, y.detach().numpy().astype(bool).transpose())})
            results.append(pd.DataFrame(res))
    results = pd.concat(results)

    pca_gnn = PCA(2).fit_transform(np.array(results['embedding'].tolist()))
    plot_odor_islands(pca_gnn, results, z_limit=0.4)
    plt.title('GNN Embeddings')
    plt.savefig(f'{fname}/pom.png')
