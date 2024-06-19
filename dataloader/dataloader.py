import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
import os
from typing import Callable, List, Optional, Union

from dataloader.representations.graphs import pyg_molecular_graphs

# Inspired by gauche DataLoader
# https://github.com/leojklarner/gauche


class DataLoader():
    """
    Loads and cleans up your data
    """

    def __init__(self):
        self.features = None
        self.labels = None

    def read_csv(self,
                 path: str,
                 smiles_column: str,
                 label_columns: List[str],
                 validate: bool = True,
                 ) -> None:
        """
        Loads a csv and stores it as features and labels.
        """
        assert isinstance(
            smiles_column, str
        ), f"smiles_column ({smiles_column}) must be a single string"
        assert isinstance(label_columns, list) and all(isinstance(item, str) for item in label_columns), "label_columns ({label_columns}) must be a list of strings."

        df = pd.read_csv(path, usecols=[smiles_column, *label_columns])
        self.features = df[smiles_column].to_numpy()
        self.labels = df[label_columns].values
        if validate:
            self.validate()

    def load_benchmark(self,
                       benchmark: str,
                       path=None,
                       validate: bool = True,
                       ) -> None:
        """
        Pulls existing benchmark from datasets.
        """
        benchmarks = {
            "leffingwell": {
                "features": "IsomericSMILES",
                # 113 labels, multiclass prediction
                "labels": ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'berry', 'black currant', 'brandy', 'bread', 'brothy', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'catty', 'chamomile', 'cheesy', 'cherry', 'chicken', 'chocolate', 'cinnamon', 'citrus', 'cocoa', 'coconut', 'coffee', 'cognac', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruity', 'garlic', 'gasoline', 'grape', 'grapefruit', 'grassy', 'green', 'hay', 'hazelnut', 'herbal', 'honey', 'horseradish', 'jasmine', 'ketonic', 'leafy', 'leathery', 'lemon', 'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'mushroom', 'musk', 'musty', 'nutty', 'odorless', 'oily', 'onion', 'orange', 'orris', 'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'pungent', 'radish', 'ripe', 'roasted', 'rose', 'rum', 'savory', 'sharp', 'smoky', 'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweet', 'tea', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 'violet', 'warm', 'waxy', 'winey', 'woody']
            },
            "mayhew_2022": {
                "features": "IsomericSMILES",
                # 1 label, odor probability prediction
                "labels": ["is_odor"],
            }
        }

        assert benchmark in benchmarks.keys(), (
            f"The specified benchmark choice ({benchmark}) is not a valid option. "
            f"Choose one of {list(benchmarks.keys())}."
        )

        # if no path is specified, use the default data directory
        if path is None:
            path = os.path.abspath(
                os.path.join(
                    os.path.abspath(__file__),
                    "..",
                    "..",
                    "datasets",
                    benchmark,
                    benchmark + "_combined.csv",
                )
            )

        self.read_csv(
            path=path,
            smiles_column=benchmarks[benchmark]["features"],
            label_columns=benchmarks[benchmark]["labels"],
            validate=validate,
        )        

    def validate(self, 
                 drop: Optional[bool] = True, 
                 canonicalize: Optional[bool] = True
    ) -> None:
        """
        Utility function to validate a read-in dataset of smiles and labels by
        checking that all SMILES strings can be converted to rdkit molecules
        and that all labels are numeric and not NaNs.
        Optionally drops all invalid entries and makes the
        remaining SMILES strings canonical (default).

        :param drop: whether to drop invalid entries
        :type drop: bool
        :param canonicalize: whether to make the SMILES strings canonical
        :type canonicalize: bool
        """

        invalid_mols = np.array(
            [
                True if MolFromSmiles(x) is None else False
                for x in self.features
            ]
        )
        if np.any(invalid_mols):
            print(
                f"Found {invalid_mols.sum()} SMILES strings "
                f"{[x for i, x in enumerate(self.features) if invalid_mols[i]]} "
                f"at indices {np.where(invalid_mols)[0].tolist()}"
            )
            print(
                "To turn validation off, use dataloader.read_csv(..., validate=False)."
            )

        invalid_labels = np.isnan(self.labels).squeeze()
        if np.any(invalid_labels):
            print(
                f"Found {invalid_labels.sum()} invalid labels "
                f"{self.labels[invalid_labels].squeeze()} "
                f"at indices {np.where(invalid_labels)[0].tolist()}"
            )
            print(
                "To turn validation off, use dataloader.read_csv(..., validate=False)."
            )
        if invalid_labels.ndim > 1:
            invalid_idx = np.any(np.hstack((invalid_mols.reshape(-1, 1), invalid_labels)), axis=1)
        else:
            invalid_idx = np.logical_or(invalid_mols, invalid_labels)

        if drop:
            self.features = [
                x for i, x in enumerate(self.features) if not invalid_idx[i]
            ]
            self.labels = self.labels[~invalid_idx]
            assert len(self.features) == len(self.labels)

        if canonicalize:
            self.features = [
                MolToSmiles(MolFromSmiles(smiles), isomericSmiles=False)
                for smiles in self.features
            ]

    def featurize(
        self, representation: Union[str, Callable], **kwargs
    ) -> None:
        """Transforms SMILES into the specified molecular representation.

        :param representation: the desired molecular representation, one of [ecfp_fingerprints, fragments, ecfp_fragprints, molecular_graphs, bag_of_smiles, bag_of_selfies, mqn] or a callable that takes a list of SMILES strings as input and returns the desired featurization.
        :type representation: str or Callable
        :param kwargs: additional keyword arguments for the representation function
        :type kwargs: dict
        """

        assert isinstance(representation, (str, Callable)), (
            f"The specified representation choice {representation} is not "
            f"a valid type. Please choose a string from the list of available "
            f"representations or provide a callable that takes a list of "
            f"SMILES strings as input and returns the desired featurization."
        )

        valid_representations = [
            "graphein_molecular_graphs",
            "pyg_molecular_graphs",
            "molecular_graphs",
        ]

        if isinstance(representation, Callable):
            self.features = representation(self.features, **kwargs)

        elif representation == "graphein_molecular_graphs":
            from .representations.graphs import graphein_molecular_graphs

            self.features = graphein_molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "pyg_molecular_graphs":
            from .representations.graphs import graphein_molecular_graphs

            self.features = pyg_molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "molecular_graphs":
            from .representations.graphs import molecular_graphs

            self.features = molecular_graphs(smiles=self.features, **kwargs)

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )
