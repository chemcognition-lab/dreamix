# sets the path to the root of the repository
from pathlib import Path
import sys
import os

root_path = os.path.dirname(os.path.abspath(""))
sys.path.append(str(root_path))
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from prediction_head.data import (
    get_regression_dataset,
    get_binary_dataset,
    get_multiclass_dataset,
    get_multilabel_dataset,
    get_zeroinflated_dataset,
    TaskType,
)

from prediction_head.GLM import GLM, train_one_epoch, train_loop, TaskSpec
from prediction_head.plot import (
    plot_regression_distribution,
    plot_classification_distribution,
    plot_ground_truth_vs_predictions,
)

# load data
dataloaders: dict = {
    TaskType.regression: get_regression_dataset(),
    TaskType.binary: get_binary_dataset(),
    TaskType.multiclass: get_multiclass_dataset(),
    TaskType.multilabel: get_multilabel_dataset(),
    TaskType.zero_inflated_binary: get_zeroinflated_dataset(
        tasktype=TaskType.zero_inflated_binary
    ),
    TaskType.zero_inflated_regression: get_zeroinflated_dataset(
        tasktype=TaskType.zero_inflated_regression
    ),
}

task_specs = [
    TaskSpec("regression", 1, TaskType.regression),
    TaskSpec("binary", 1, TaskType.binary),
    TaskSpec("multiclass", 5, TaskType.multiclass),
    TaskSpec("multilabel", 5, TaskType.multilabel),
    # TaskSpec("zero_inflated_binary", 1, TaskType.zero_inflated_binary),
    # TaskSpec("zero_inflated_regression", 1, TaskType.zero_inflated_regression),
]
# run ML model
results = train_loop(25, task_specs, dataloaders=dataloaders, epochs=10)

# print results but only the first item in the dictionary
for key, value in results.items():
    print(f"{key=}, {value[0]=}")
