from typing import Iterable, Union, Optional, Callable, List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as pygdl

import numpy as np


class DreamDataset(Dataset):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = inputs
        self.labels = labels

        self.representation_dim = self.inputs.shape[2]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        labels = self.labels[idx]
        return input_data, labels

def map_nested_list(nested_list, mapping_dict):
    def map_element(x):
        return mapping_dict.get(x, x)
    
    def recursive_map(item):
        if isinstance(item, list):
            return [recursive_map(sub_item) for sub_item in item]
        else:
            return map_element(item)
    
    return recursive_map(nested_list)


def get_mixture_embeddings( 
        mixtures: np.ndarray, 
        from_smiles: Callable,
        pad_idx: Optional[int] = -999,
    ):
    # from_smiles function should create vector from smiles
    flat_mix = mixtures.flatten()
    pad_len = max([len(x) for x in flat_mix])

    mixtures_processed = []
    for mix in mixtures.transpose():
        smiles_processed = []
        for smi_arr in mix:
            smiles_processed.append(np.pad(smi_arr, (0, pad_len - len(smi_arr)), constant_values=''))
        mixtures_processed.append(smiles_processed)
    mixtures_processed = np.array(mixtures_processed).transpose((1,2,0)).tolist()

    # map it to the feature
    smiles_list = list(set([x for xs in flat_mix for x in xs]))
    feature_map = {smi: from_smiles(smi) for smi in smiles_list}
    feat = feature_map[smiles_list[0]]
    feature_map[''] = np.full(feat.shape, fill_value=pad_idx)
        
    mixtures = map_nested_list(mixtures_processed, feature_map)
    mixtures = np.array(mixtures).transpose((0, 1, 3, 2))
    return mixtures


def get_mixture_smiles(
        mixtures: np.ndarray, 
        from_smiles: Callable
    ) -> Union[List[Data], np.ndarray]:
    # this function will return the graphs for all smiles
    # present in a mixture, returning them as a list
    flat_mix = mixtures.flatten()
    smiles_list = list(set([x for xs in flat_mix for x in xs]))

    pad_len = max([len(x) for x in flat_mix])
    mixtures_processed = []
    for mix in mixtures.transpose():
        smiles_processed = []
        for smi_arr in mix:
            smiles_processed.append(np.pad(smi_arr, (0, pad_len - len(smi_arr)), constant_values=''))
        mixtures_processed.append(smiles_processed)
    mixtures_processed = np.array(mixtures_processed).transpose((1,2,0)).tolist()

    # map it to the feature
    smiles_list = list(set([x for xs in flat_mix for x in xs]))
    feature_map = {smi: i-1 for i, smi in enumerate([''] + smiles_list)}
    mix_inds = map_nested_list(mixtures_processed, feature_map)

    return [from_smiles(smi) for smi in smiles_list], np.array(mix_inds) 
    

class MixtureVectorDataset(Dataset):
    def __init__(self, 
                 mixture_vectors: np.ndarray, 
                 labels: Iterable[Union[float,int]],
                 pad_idx: Optional[int] = -999,
                ):
        super().__init__()
        self.mixtures = mixture_vectors
        self.labels = labels
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.mixtures)
    
    def __getitem__(self, idx):
        return torch.tensor(self.mixtures[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)   
