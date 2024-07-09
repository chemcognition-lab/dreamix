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
    TaskType.multiclass: get_multiclass_dataset(n_features=25, n_classes=5)
}
task_specs = [TaskSpec(TaskType.multiclass, 5, TaskType.multiclass)]
train_dataloader = dataloaders[TaskType.multiclass][0]
test_dataloader = dataloaders[TaskType.multiclass][1]
scaler = dataloaders[TaskType.multiclass][2]
# run ML model
results = train_loop(25, task_specs, dataloaders, epochs=10)
print(results["multiclass"][0])
