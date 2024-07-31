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
from dataloader import DreamLoader

# pom
import pom.utils as utils
from pom import EndToEndModule
from pom.data import GraphDataset
from pom.early_stop import EarlyStopping
from pom.gnn.graphnets import GraphNets

# glm
from prediction_head.data import TaskType, get_loss_fn
from prediction_head.GLM import GLMStructured, TaskSpec

import torch
import torchinfo
from torch_geometric.loader import DataLoader as pygdl

from argparse import ArgumentParser


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

def get_split_file(dataset_name, test_size):
    fname = f'{dataset_name}_models/{dataset_name}_{test_size}.npz'
    if os.path.isfile(fname):
        split = np.load(fname)
        train_ind, test_ind = split['train_ind'], split['test_ind']
    else:
        dl = DreamLoader()
        dl.load_benchmark(dataset_name)
        num_dat = dl.get_dataset_specifications(dataset_name)['n_datapoints']
        labels = dl.labels
        train_ind, _, test_ind, _ = iterative_train_test_split(np.array(range(num_dat)).reshape(-1,1), 
                                                               labels, 
                                                               test_size=test_size)
        np.savez(fname, train_ind=train_ind, test_ind=test_ind)
    return train_ind, test_ind


def get_split_file(dataset_name, test_size):
    os.makedirs(f'{dataset_name}_models/', exist_ok=True)
    fname = f'{dataset_name}_models/{dataset_name}_{test_size}.npz'
    
    if os.path.isfile(fname):
        split = np.load(fname)
        train_ind, test_ind = split['train_ind'], split['test_ind']
    else:
        dl = DreamLoader()
        dl.load_benchmark(dataset_name)
        labels = dl.labels
        num_dat = len(labels)
        train_ind, _, test_ind, _ = iterative_train_test_split(np.array(range(num_dat)).reshape(-1,1), 
                                                               labels, 
                                                               test_size=test_size)
        np.savez(fname, train_ind=train_ind, test_ind=test_ind)
    return train_ind, test_ind


parser = ArgumentParser()
parser.add_argument("--dir", action="store", type=str, default='general_models/model1', help="Name of directory of POM.")
FLAGS = parser.parse_args()


if __name__ == '__main__':
    # read the hyperparameters
    fname = FLAGS.dir
    hp = ConfigDict(json.load(open(f'{fname}/hparams.json')))

    # get dataset names
    seed = 42
    utils.set_seed(seed)
    data_names =  ['gs-lf']     # use this dataset, so that we can look at the labels

    # Load all models and datasets listed
    print(f'Using GPU: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    specs = {}
    data_store = {}
    for dname in data_names:
        print(f'Loading {dname}...')
        dl = DreamLoader()
        dl.load_benchmark(dname)
        dl.featurize('molecular_graphs', init_globals=True)
        data_specs = dl.get_dataset_specifications(dname)

        # Dataset specifics
        task = TaskType(data_specs['task'])
        task_dim = data_specs['task_dim']
        task_spec = TaskSpec(task_dim, task)
        specs[dname] = task_spec

        train_ind, test_ind = get_split_file(dname, test_size=0.2)
        if dname == 'keller_2016':
            dl.labels /= 100.
        dataset = GraphDataset(dl.features, dl.labels)

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
                    'task': task,
                }
            } 
        )

    # create model
    gnn = GraphNets(
        dataset.node_dim, 
        dataset.edge_dim, 
        hp.global_dim, 
        hidden_dim = hp.hidden_dim, 
        depth=hp.depth, 
        dropout = hp.dropout
    ).to(device)
    pred = GLMStructured(input_dim=hp.global_dim, tasks=specs).to(device)
    model = EndToEndModule(gnn, pred).to(device)

    # draw the POM
    model.gnn_embedder.load_state_dict(torch.load(f'{fname}/gnn_embedder.pt'))      # load the best one trained
    model = model.to(torch.device('cpu'))       # doing it on the CPU
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

    pca = PCA(2)
    pca_gnn = pca.fit_transform(np.array(results['embedding'].tolist()))
    pca_ex = pca.explained_variance_ratio_[:2]
    plot_odor_islands(pca_gnn, results, z_limit=0.4)
    plt.xlabel(f'PCA1 ({pca_ex[0]*100:.2f}%)')
    plt.ylabel(f'PCA2 ({pca_ex[1]*100:.2f}%)')
    plt.title('GNN Embeddings')
    plt.savefig(f'{fname}/pom.png')


