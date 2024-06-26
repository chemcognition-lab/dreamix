from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from rdkit.Chem import MolFromSmiles
from mordred import Calculator, descriptors
from typing import List, Optional
import numpy as np
import logging
import pandas as pd

def parse_status(generator, smiles):
    results = generator.process(smiles)
    try: 
        processed, features = results[0], results[1:]
        if processed is None:
            logging.warning("Descriptastorus cannot process smiles %s", smiles)
        return features
    except TypeError:
        logging.warning("RDKit Error on smiles %s", smiles)
        # if processed is None, the features are are default values for the type


def morgan_fingerprints(
    smiles: List[str],
) -> np.ndarray:
    """
    Builds molecular representation as a binary Morgan ECFP fingerprints with radius 3 and 2048 bits.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: array of shape [len(smiles), 2048] with ecfp featurised molecules

    """
    generator = MakeGenerator((f"Morgan3",))
    fps = np.array([parse_status(generator, x) for x in smiles])
    return fps

def rdkit2d_normalized_features(
    smiles: List[str],
) -> np.ndarray:
    """
    Builds molecular representation as normalized 2D RDKit features.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: array of shape [len(smiles), 200] with featurised molecules

    """
    generator = MakeGenerator((f"rdkit2dhistogramnormalized",))
    fps = np.array([parse_status(generator, x) for x in smiles])
    return fps

def rowwise_aggregate_mixture_rdkit2d_normalized_features(
        mixture_smiles: List[List[str]],
) -> np.ndarray:
    """
    Builds molecular representation as aggregated normalized 2D RDKit features for a mixture.
    The mixture is provided as a list of 114 smiles. The first half corresponds to the first mixture.
    The aggregation is done per mixture.

    :param mixture_smiles: list of smiles within the mixture
    :type mixture_smiles: list
    :return: array of shape [1600, ] with featurised mixture

    """
    def get_aggregate(mixture):
        return np.array([np.mean(mixture, axis=0), np.var(mixture, axis=0), np.max(mixture, axis=0), np.min(mixture, axis=0)]).flatten()

    generator = MakeGenerator((f"rdkit2dhistogramnormalized",))
    mixtures = np.split(np.array(mixture_smiles),2)
    mixtures[0] = mixtures[0][~pd.isnull(mixtures[0])]
    mixtures[1] = mixtures[1][~pd.isnull(mixtures[1])]
    aggregate_0 = get_aggregate([parse_status(generator, smiles) for smiles in mixtures[0]])
    aggregate_1 = get_aggregate([parse_status(generator, smiles) for smiles in mixtures[1]])
    aggregate = np.concatenate((aggregate_0, aggregate_1), axis=0)
    return aggregate

def mordred_descriptors(    
    smiles: List[str],
) -> np.ndarray:
    """
    Calculates Mordred descriptors.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: array of shape [len(smiles), 1613] with featurised molecules

    """
    calc = Calculator(descriptors, ignore_3D=True)
    # TODO: invalid smiles handling
    mols = np.array([MolFromSmiles(smi) for smi in smiles])
    features = np.array([calc(mol) for mol in mols])
    return features