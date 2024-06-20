from dataloader import DataLoader
from xgboost import XGBClassifier
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
from torchmetrics.functional.classification import multilabel_auroc
import torch

data = DataLoader()
data.load_benchmark("leffingwell")
data.featurize("rdkit2d_normalized_features")

X = data.features
y = data.labels

num_class = y.shape[1]

X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)
bst = XGBClassifier(tree_method="hist",
                        n_estimators=500, max_depth=10, learning_rate=0.1,
                        early_stopping_rounds=10, eval_metric='logloss', n_jobs=-1)
bst.fit(X_train, y_train,eval_set=[(X_test, y_test)])
y_pred = bst.predict(X_test)

yt_pred = torch.from_numpy(y_pred)
yt_test = torch.from_numpy(y_test)
auroc = multilabel_auroc(preds=yt_pred, target=yt_test, num_labels=y.shape[1])

print(auroc)


