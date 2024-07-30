import copy
from pprint import pprint

import numpy as np
import omegaconf
import torch

import torchmetrics.functional as F
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau, spearmanr, pearsonr

def print_conf(conf: omegaconf.DictConfig):
    """Pretty prints a configuration object."""
    pprint(omegaconf.OmegaConf.to_container(conf, resolve=True))


class EarlyStopping:
    """Stop training early if a metric stops improving.

    Models often benefit from stoping early after a metric stops improving.
    This implementation assumes the monitored value will be loss-like
    (i.g. val_loss) and will checkpoint when reaching a new best value.
    Checkpointed value can be restored.

    Args:
      model: model to checkpoint.
      patience: number of iterations before flaggin a stop.
      min_delta: minimum value to quanlify as an improvement.
      checkpoint_interval: number of iterations before checkpointing.
      mode: maximise or minimise the monitor value
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patience: int = 100,
        min_delta: float = 0,
        checkpoint_interval: int = 1,
        mode: bool = "maximize",
    ):
        self.patience = patience
        self.min_delta = np.abs(min_delta)
        self.wait = 0
        self.best_step = 0
        self.checkpoint_count = 0
        self.checkpoint_interval = checkpoint_interval
        self.values = []
        self.best_model = copy.deepcopy(model.state_dict())
        if mode == "maximize":
            self.monitor_op = lambda a, b: np.greater_equal(a - min_delta, b)
            self.best_value = -np.inf
        elif mode == "minimize":
            self.monitor_op = lambda a, b: np.less_equal(a + min_delta, b)
            self.best_value = np.inf
        else:
            raise ValueError("Invalid mode for early stopping.")

    def check_criteria(self, monitor_value: float, model: torch.nn.Module) -> bool:
        """Gets learing rate based on value to monitor."""
        self.values.append(monitor_value)
        self.checkpoint_count += 1

        if self.monitor_op(monitor_value, self.best_value):
            self.best_value = monitor_value
            self.best_step = len(self.values) - 1
            self.wait = 0
            if self.checkpoint_count >= self.checkpoint_interval:
                self.checkpoint_count = 0
                self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1

        return self.wait >= self.patience

    def restore_best(self):
        print(
            f"Restoring checkpoint at step {self.best_step} with best value at {self.best_value}"
        )
        return self.best_model

TORCH_METRIC_FUNCTIONS = {
    'pearson': F.pearson_corrcoef,
    'spearman': F.spearman_corrcoef,
    'kendall': F.kendall_rank_corrcoef,
    'r2': F.r2_score,
    'rmse': lambda pred, targ: F.mean_squared_error(pred, targ, squared=False),
    'mae': F.mean_absolute_error
}

NUMPY_METRIC_FUNCTIONS = {
    'pearson': lambda x, y: pearsonr(x,y)[0],
    'spearman': lambda x, y: spearmanr(x,y)[0],
    'kendall': lambda x, y: kendalltau(x,y)[0],
    'r2': r2_score,
    'rmse': root_mean_squared_error,
    'mae': mean_absolute_error
}
