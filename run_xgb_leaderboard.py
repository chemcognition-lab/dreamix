from dataloader import DataLoader
from xgboost import XGBClassifier, XGBRegressor
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau
import numpy as np
import pandas as pd
import sys

data = DataLoader()
data.load_benchmark("competition_train")
#data.featurize("competition_rdkit2d_augment")
data.featurize("competition_rdkit2d")

X = data.features
y = data.labels
converged = False

while not converged:
    if False:
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)
    else:
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)
        X_train = np.vstack((X_train.reshape(400,1600), np.array([[x[1], x[0]] for x in X_train]).reshape(400,1600)))
        X_test = np.vstack((X_test.reshape(100,1600), np.array([[x[1], x[0]] for x in X_test]).reshape(100,1600)))
        y_train = np.vstack((y_train, y_train))
        y_test = np.vstack((y_test, y_test))

    bst = XGBRegressor(n_estimators=160000, max_depth=1000, learning_rate=0.15, subsample=0.8, colsample_bytree=0.8,
                                early_stopping_rounds=10, eval_metric=mean_squared_error, n_jobs=-1, random_state=42)

    bst.fit(X_train, y_train,eval_set=[(X_test, y_test)])
    y_pred = bst.predict(X_test)


    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'Root Mean Squared Error: {rmse}')

    r2 = r2_score(y_test, y_pred)
    print(f'R-squared: {r2}')

    prs = pearsonr(y_test.flatten(), y_pred.flatten())
    print(f'Pearson coefficient: {prs}')

    ket = kendalltau(y_test, y_pred)
    print(f'Kendall Tau: {ket}')
    
    if prs[0] >= float(sys.argv[1]):
        converged=True


lb_data = DataLoader()
lb_data.load_benchmark("competition_leaderboard")
lb_data.featurize("competition_rdkit2d")


X_lb = lb_data.features.reshape(46, 1600)

y_lb = bst.predict(X_lb)

pd.DataFrame(y_lb, columns=['prediction']).to_csv("leaderboard.csv", index=False)
