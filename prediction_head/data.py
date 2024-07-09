# sets the path to the root of the repository
from pathlib import Path
import sys

root_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(root_path))

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import enum
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
)
from scipy.stats import kendalltau, spearmanr
from prediction_head.loss import FocalLoss
import functools

# Load Data (Create Dataset and Dataloader)
from torch.utils.data import DataLoader, Dataset

np.random.seed(2)


class TaskType(enum.StrEnum):
    regression = enum.auto()
    binary = enum.auto()
    multiclass = enum.auto()
    multilabel = enum.auto()
    zero_inflated_binary = enum.auto()
    zero_inflated_regression = enum.auto()
    zero_inflated_holistic = enum.auto()


def get_activation(tasktype: TaskType) -> nn.Module:
    activations = {
        TaskType.regression: nn.Identity,
        TaskType.binary: nn.Identity,
        TaskType.multiclass: nn.Identity,  # functools.partial(nn.Softmax, dim=1),
        TaskType.multilabel: nn.Identity,
    }
    return activations[tasktype]


def get_loss_fn(tasktype: TaskType):
    loss_fns: dict = {
        TaskType.regression: nn.MSELoss,
        TaskType.binary: nn.BCEWithLogitsLoss,      # requires logits
        TaskType.multiclass: nn.CrossEntropyLoss,   # requires logits
        TaskType.multilabel: nn.BCEWithLogitsLoss,  # requires logits
        TaskType.zero_inflated_binary: nn.BCELoss,  # requires probabilities
        TaskType.zero_inflated_regression: nn.MSELoss,
    }
    return loss_fns[tasktype]


def get_metrics(tasktype: TaskType):
    metrics: dict[list] = {
        TaskType.regression: [
            r2_score,
            np.corrcoef,
            root_mean_squared_error,
            mean_absolute_error,
        ],
        TaskType.binary: [roc_auc_score, accuracy_score],
        TaskType.multiclass: [
            functools.partial(roc_auc_score, multi_class="ovr"),
            accuracy_score,
        ],
        TaskType.multilabel: [
            functools.partial(roc_auc_score, multi_class="ovr"),
            accuracy_score,
        ],
        TaskType.zero_inflated_binary: [roc_auc_score, accuracy_score],
        TaskType.zero_inflated_regression: [
            r2_score,
            np.corrcoef,
            root_mean_squared_error,
            mean_absolute_error,
        ],
        TaskType.zero_inflated_holistic: [kendalltau, spearmanr],
    }
    metric_names: dict[list] = {
        TaskType.regression: ["r2", "r", "rmse", "mae"],
        TaskType.binary: ["roc", "acc"],
        TaskType.multiclass: ["roc", "acc"],
        TaskType.multilabel: ["roc", "acc"],
        TaskType.zero_inflated_binary: ["roc", "acc"],
        TaskType.zero_inflated_regression: ["r2", "r", "rmse", "mae"],
        TaskType.zero_inflated_holistic: ["kendalltau", "spearmanr"],
    }
    return zip(metrics[tasktype], metric_names[tasktype])


def get_optimizer(tasktype: TaskType):
    return torch.optim.Adam


def min_max_scale(train_data, test_data):
    scaler = MinMaxScaler()
    x_scaler = scaler.fit(train_data)
    train_data = x_scaler.transform(train_data)
    test_data = x_scaler.transform(test_data)
    return train_data, test_data, scaler


class SyntheticSklearnDatasets(Dataset):
    def __init__(self, x, y, y_mask):
        self.x: torch.tensor = x
        self.y: torch.tensor = y
        self.y_mask: torch.tensor = y_mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.y_mask[idx]


def get_regression_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    noise=0.1,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=22,
    )

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)
    y_train, y_test, scaler = min_max_scale(
        y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    )

    # get mask for regression dataset which does not mask anything
    y_train_mask = np.ones_like(y_train)
    y_test_mask = np.ones_like(y_test)

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    train_dataset = SyntheticSklearnDatasets(x_train, y_train, y_train_mask)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test, y_test_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader, scaler, y_train_mask, y_test_mask


def get_binary_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    n_redundant: int = 2,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=22,
    )

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    # get mask for binary dataset which does not mask anything
    y_train_mask = np.ones_like(y_train)
    y_test_mask = np.ones_like(y_test)

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    y_train_mask = torch.from_numpy(y_train_mask)
    y_test_mask = torch.from_numpy(y_test_mask)
    train_dataset = SyntheticSklearnDatasets(x_train, y_train, y_train_mask)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test, y_test_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader, scaler, y_train_mask, y_test_mask


def get_multilabel_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_classes: int = 5,
    n_labels: int = 3,
    allow_unlabeled: bool = False,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        random_state=22,
        allow_unlabeled=allow_unlabeled,
    )

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    # get mask for multilabel dataset which does not mask anything
    y_train_mask = np.ones_like(y_train)
    y_test_mask = np.ones_like(y_test)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train, y_train_mask)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test, y_test_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader, scaler, y_train_mask, y_test_mask


def get_multiclass_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_classes: int = 5,
    n_informative: int = 10,
    n_redundant: int = 2,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        n_redundant=n_redundant,
        random_state=22,
    )

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    # get mask for multilabel dataset which does not mask anything
    y_train_mask = np.ones_like(y_train)
    y_test_mask = np.ones_like(y_test)
    train_dataset = SyntheticSklearnDatasets(x_train, y_train, y_train_mask)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test, y_test_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader, scaler, y_train_mask, y_test_mask


def get_zeroinflated_dataset(
    tasktype: TaskType,
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    noise: float = 0.1,
    zero_inflated_percent: float = 0.8,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=22,
    )

    # Add zero-inflated datapoints
    y = np.abs(y)
    # Set the y-value of 80% of the sample to 0
    is_zero = np.random.choice(
        y.shape[0], int(y.shape[0] * zero_inflated_percent), replace=False
    )
    y[is_zero] = 0
    # Convert non-zero datapoints to 1, //
    # and mask the zero-inflated datapoints if the tasktype is zero-inflated-regression
    if tasktype == TaskType.zero_inflated_binary:
        y = np.where(y != 0, 1, 0)
        # Pre-processing
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # Splitting the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=22
        )
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
        # min-max scale
        x_train, x_test, scaler = min_max_scale(x_train, x_test)

        # get mask for binary dataset which does not mask anything
        y_train_mask = np.ones_like(y_train)
        y_test_mask = np.ones_like(y_test)

    elif tasktype == TaskType.zero_inflated_regression:
        # Pre-processing
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # Splitting the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=22
        )

        # min-max scale
        x_train, x_test, scaler = min_max_scale(x_train, x_test)
        y_train, y_test, scaler = min_max_scale(
            y_train.reshape(-1, 1), y_test.reshape(-1, 1)
        )

        # get mask for zero-inflated-regression dataset which masks all zero values
        y_train_mask = torch.tensor(torch.tensor(np.where(y_train == 0, 0, 1)))
        y_test_mask = torch.tensor(torch.tensor(np.where(y_test == 0, 0, 1)))

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train, y_train_mask)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test, y_test_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader, scaler, y_train_mask, y_test_mask


def get_zeroinflated_negativebinomial_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    noise: float = 0.1,
    zero_inflated_percent: float = 0.8,
    tasktype: TaskType = TaskType.zero_inflated_regression,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=22,
    )

    y = np.random.default_rng(seed=22).negative_binomial(size=n_samples, n=1, p=0.5)

    # Add zero-inflated datapoints by making zero_inflated_percent number of datapoints equal to 0
    is_zero = np.random.choice(
        y.shape[0], int(y.shape[0] * zero_inflated_percent), replace=False
    )
    y[is_zero] = 0

    # One-hot-encode the zero-inflated datapoints if the tasktype is zero-inflated-binary, //
    # and mask the zero-inflated datapoints if the tasktype is zero-inflated-regression
    if tasktype == TaskType.zero_inflated_binary:
        y = np.where(y == 0, 1, 0)
        # Pre-processing
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # Splitting the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=22
        )
        # get mask for binary dataset which does not mask anything
        y_train_mask = np.ones_like(y_train)
        y_test_mask = np.ones_like(y_test)
        # min-max scale
        x_train, x_test, scaler = min_max_scale(x_train, x_test)

    elif tasktype == TaskType.zero_inflated_regression:
        # Pre-processing
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # Splitting the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=22
        )

        # min-max scale
        x_train, x_test, scaler = min_max_scale(x_train, x_test)
        y_train, y_test, scaler = min_max_scale(
            y_train.reshape(-1, 1), y_test.reshape(-1, 1)
        )

        # get mask for zero-inflated-regression dataset which masks all zero values
        y_train_mask = torch.tensor(np.where(y_train == 0, 0, 1))
        y_test_mask = torch.tensor(np.where(y_test == 0, 0, 1))

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train, y_train_mask)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test, y_test_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader, scaler, y_train_mask, y_test_mask


def get_zeroinflated_exponential_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    noise: float = 0.1,
    zero_inflated_percent: float = 0.8,
    tasktype: TaskType = TaskType.zero_inflated_regression,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=22,
    )

    y = np.random.default_rng(seed=22).exponential(scale=4, size=n_samples)

    # Add zero-inflated datapoints by making zero_inflated_percent number of datapoints equal to 0
    is_zero = np.random.choice(
        y.shape[0], int(y.shape[0] * zero_inflated_percent), replace=False
    )
    y[is_zero] = 0

    # One-hot-encode the zero-inflated datapoints if the tasktype is zero-inflated-binary, //
    # and mask the zero-inflated datapoints if the tasktype is zero-inflated-regression
    if tasktype == TaskType.zero_inflated_binary:
        y = np.where(y == 0, 1, 0)
        # Pre-processing
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # Splitting the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=22
        )
        # get mask for binary dataset which does not mask anything
        y_train_mask = np.ones_like(y_train)
        y_test_mask = np.ones_like(y_test)
        # min-max scale
        x_train, x_test, scaler = min_max_scale(x_train, x_test)

    elif tasktype == TaskType.zero_inflated_regression:
        # Pre-processing
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # Splitting the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=22
        )

        # min-max scale
        x_train, x_test, scaler = min_max_scale(x_train, x_test)
        y_train, y_test, scaler = min_max_scale(
            y_train.reshape(-1, 1), y_test.reshape(-1, 1)
        )

        # get mask for zero-inflated-regression dataset which masks all zero values
        y_train_mask = torch.tensor(np.where(y_train == 0, 0, 1))
        y_test_mask = torch.tensor(np.where(y_test == 0, 0, 1))

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train, y_train_mask)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test, y_test_mask)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader, scaler, y_train_mask, y_test_mask
