import sys
sys.path.append('..')

from dataloader import DataLoader
from xgboost import XGBClassifier, XGBRegressor
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau
import numpy as np
#from torchmetrics.functional.classification import multilabel_auroc
import torch
import pandas as pd

data = DataLoader()
#data.load_benchmark("leffingwell")
#data.featurize("rdkit2d_normalized_features")
data.load_benchmark("competition_train")
data.featurize("competition_rdkit2d")

X = data.features.reshape(500, 1600)
y = data.labels

X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)

if False:
    bst = XGBClassifier(tree_method="hist",
                            n_estimators=500, max_depth=10, learning_rate=0.1,
                            early_stopping_rounds=10, eval_metric='logloss', n_jobs=-1)
else:
    bst = XGBRegressor(n_estimators=16000, max_depth=100, learning_rate=0.2,
                            early_stopping_rounds=10, eval_metric=mean_squared_error, n_jobs=-1, random_state=42)

bst.fit(X_train, y_train,eval_set=[(X_test, y_test)])
y_pred = bst.predict(X_test)


if False:
    yt_pred = torch.from_numpy(y_pred)
    yt_test = torch.from_numpy(y_test)
    auroc = multilabel_auroc(preds=yt_pred, target=yt_test, num_labels=y.shape[1])
    print(auroc)
else:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'Root Mean Squared Error: {rmse}')

    r2 = r2_score(y_test, y_pred)
    print(f'R-squared: {r2}')

    prs = pearsonr(y_test.flatten(), y_pred.flatten())
    print(f'Pearson coefficient: {prs}')

    ket = kendalltau(y_test, y_pred)
    print(f'Kendall Tau: {ket}')


lb_data = DataLoader()
lb_data.load_benchmark("competition_leaderboard")
lb_data.featurize("competition_rdkit2d")


X_lb = lb_data.features.reshape(46, 1600)

y_lb = bst.predict(X_lb)

pd.DataFrame(y_lb, columns=['prediction']).to_csv("leaderboard.csv", index=False)