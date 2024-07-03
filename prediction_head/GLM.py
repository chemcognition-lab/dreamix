from __future__ import annotations

# sets the path to the root of the repository
from pathlib import Path
import sys

from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

root_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(root_path))
from prediction_head.data import (
    TaskType,
    get_activation,
    get_loss_fn,
    get_metrics,
    get_optimizer,
)
from torch import nn
import numpy as np
import torch
import dataclasses
import functools


@dataclasses.dataclass
class TaskSpec:
    "Task specification for a GLM."
    name: str
    dim: int
    task: TaskType


class GLM(nn.Module):
    "Generalized Linear Model for basic predictive modelling."

    def __init__(self, input_dim: int, output_dim: int, tasktype: TaskType):
        super(GLM, self).__init__()
        self.tasktype = tasktype
        self.activation = get_activation(tasktype)()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.activation(self.linear(x))

    @classmethod
    def from_spec(cls, input_dim: int, spec: TaskSpec) -> GLM:
        return cls(
            input_dim=input_dim,
            output_dim=spec.dim,
            tasktype=spec.task,
        )


class GLMStructured(nn.Module):
    "Generalized Linear Models for predicting multiple outputs."

    def __init__(self, input_dim: int, tasks: list[TaskSpec]):
        self.models = {task.name: GLM.from_spec(input_dim, task) for task in tasks}

    def forward(self, x):
        return {name: model(x) for name, model in self.models.items()}


def get_probability_calibration(predictions, y_test, metric) -> float:
    "Calibrate the probability threshold to the value that gives you the best value of a given metric."
    threshold = np.arange(0, 1, 0.01)
    scores = []
    for t in threshold:
        predictions_binary = np.where(predictions > t, 1, 0)
        scores.append(metric(y_test, predictions_binary))
    best_threshold = threshold[np.argmax(scores)]
    return best_threshold


def train_one_epoch(
    epoch_index: int, optimizer, dataloader, loss_fn, model, n_batches: int = 10
):
    running_loss = 0.0
    batch_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs, labels, y_mask = data
        optimizer.zero_grad()
        outputs = model(inputs)

        # def loss_fn(outputs, labels, mask):
        #     return loss_fn(outputs, labels)

        # def loss_fn(outputs, labels, mask):
        #     binary_loss = nn.BCELoss()(outputs > 0, labels)
        #     reg_loss = nn.MSELoss()(outputs * mask, labels * mask)
        #     return binary_loss + reg_loss

        # loss = loss_fn(outputs, y_mask, labels, y_mask)  # mask out zero values
        # loss = loss_fn(outputs * y_mask, labels * y_mask)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % n_batches == n_batches - 1:
            batch_loss = running_loss / n_batches
            running_loss = 0.0

    return batch_loss


def train_loop(
    input_dim: int,
    task_specs: dict,
    dataloaders: dict,
    epochs: int,
) -> dict:
    glm_structured = GLMStructured(input_dim, task_specs)
    glm_structured_results = {}
    for (name, model), dataloader in zip(
        glm_structured.models.items(),
        dataloaders.values(),
    ):
        print(f"{name=}")
        epoch_number: int = 0

        best_vloss: float = 1_000_000.0

        loss_fn = get_loss_fn(model.tasktype)()
        optimizer = get_optimizer(model.tasktype)(model.parameters(), lr=1e-3)
        train_dataloader = dataloader[0]
        test_dataloader = dataloader[1]
        scaler = dataloader[2]
        global_y_train_mask = dataloader[3]
        global_y_test_mask = dataloader[4]

        for epoch in range(epochs):
            # print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch(
                epoch_number, optimizer, train_dataloader, loss_fn, model, 10
            )
            # print(f"{avg_loss=}")

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(test_dataloader):
                    vinputs, vlabels, y_test_mask = vdata
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    # vloss = loss_fn(voutputs * y_test_mask, vlabels * y_test_mask)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss, ".5f"))

            # Log the running loss averaged per batch
            # for both training and validation

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                # torch.save(model.state_dict(), model_path)

            epoch_number += 1
        # Overall Test Score
        predictions = model(test_dataloader.dataset.x).detach().numpy()
        y_test = test_dataloader.dataset.y.detach().numpy()
        if (
            model.tasktype == TaskType.regression
            or model.tasktype == TaskType.zero_inflated_regression
        ):
            predictions = scaler.inverse_transform(predictions)
            y_test = scaler.inverse_transform(y_test)
        elif (
            model.tasktype == TaskType.binary
            or model.tasktype == TaskType.zero_inflated_binary
        ):
            calibration_threshold = get_probability_calibration(
                predictions, y_test, f1_score
            )
            # convert predictions to binary
            predictions = np.where(predictions > calibration_threshold, 1, 0)
        elif model.tasktype == TaskType.multiclass:
            ohe = OneHotEncoder()
            y_test = ohe.fit_transform(y_test.reshape(-1, 1)).toarray()
        elif model.tasktype == TaskType.multilabel:
            calibration_threshold = get_probability_calibration(
                predictions, y_test, functools.partial(f1_score, average="micro")
            )
            # convert predictions to binary
            predictions = np.where(predictions > calibration_threshold, 1, 0)
        # Aggregate results
        results = {}
        for metric, metric_name in get_metrics(model.tasktype):
            if metric_name == "r":
                results[metric_name] = metric(y_test.flatten(), predictions.flatten())[
                    0, 1
                ]
            elif metric_name == "acc":
                # convert predictions to one-hot-encoding
                ones = np.zeros_like(y_test)
                predictions = np.argmax(predictions, axis=-1)
                ones[np.arange(len(predictions)), predictions] = 1
                results[metric_name] = metric(y_test, ones)
            else:
                results[metric_name] = metric(y_test, predictions)
        glm_structured_results[name] = [results, y_test, predictions]
    if (
        "zero_inflated_binary" in glm_structured_results.keys()
        and "zero_inflated_regression" in glm_structured_results.keys()
    ):
        # gather y_test and predictions from zero inflated binary and regression models
        # convert one-hot-encodings back to binary
        y_test_binary = glm_structured_results["zero_inflated_binary"][1]
        y_test_regression = glm_structured_results["zero_inflated_regression"][1]
        predictions_binary = glm_structured_results["zero_inflated_binary"][2]
        predictions_regression = glm_structured_results["zero_inflated_regression"][2]
        # replace non-zero values with the regression values for both y_test and predictions
        y_test_zero_inflated = np.where(y_test_binary == 0)[1].reshape(-1, 1)
        y_test_zero_inflated = np.where(
            y_test_zero_inflated == 1, y_test_regression, y_test_zero_inflated
        )
        predictions_zero_inflated = np.where(predictions_binary == 0)[1].reshape(-1, 1)
        predictions_zero_inflated = np.where(
            predictions_zero_inflated == 1,
            predictions_regression,
            predictions_zero_inflated,
        )
        # print(f"{y_test_zero_inflated=}, {predictions_zero_inflated=}")
        # add kendalltau and spearmanr to zero_inflated_binary
        results = {}
        for metric, metric_name in get_metrics(TaskType.zero_inflated_holistic):
            results[metric_name] = metric(
                y_test_zero_inflated, predictions_zero_inflated
            )
        glm_structured_results["zero_inflated_holistic"] = [
            results,
            y_test_zero_inflated,
            predictions_zero_inflated,
        ]

    return glm_structured_results
