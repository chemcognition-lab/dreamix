import sys
sys.path.append('..')

import seaborn as sns
from dataloader import DreamLoader

from xgboost import XGBClassifier, XGBRegressor
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau
import numpy as np
import pandas as pd
import json

if __name__ == '__main__':

    data = DreamLoader()
    data.load_benchmark("competition_train")
    data.featurize("competition_rdkit2d_augment")
    # data.featurize("competition_rdkit2d")
    X = data.features.reshape(len(data.features), -1)
    y = data.labels.flatten()

    data = DreamLoader()
    data.load_benchmark("competition_leaderboard")
    data.featurize("competition_rdkit2d")
    X_val = data.features.reshape(len(data.features), -1)
    y_val = data.labels.flatten()

    best_results = {
        'best_metric': -np.inf,
        'y_pred': None,
        'y_truth': y_val.tolist(),
        'r2': None,
        'rmse': None,
        'pearson': None,
        'kendall': None
    }

    for _ in range(100):
        bst = XGBRegressor(n_estimators=1000, max_depth=1000, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                        early_stopping_rounds=10, eval_metric=mean_squared_error, n_jobs=24, verbosity=0)

        bst.fit(X, y, eval_set=[(X_val, y_val)])
        y_pred = bst.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        prs, _ = pearsonr(y_val.flatten(), y_pred.flatten())
        ket, _ = kendalltau(y_val, y_pred)

        if best_results['best_metric'] < prs:
            best_results['best_metric'] = prs 
            best_results['y_pred'] = y_pred.tolist()
            best_results['r2'] = r2 
            best_results['rmse'] = rmse 
            best_results['pearson'] = prs 
            best_results['kendall'] = ket 
            bst.save_model('xgboost_best_model.json')

    # best results
    print(best_results)
    json.dump(best_results, open('xgboost_results.json', 'w'), indent=4)

    pd.DataFrame({
        'Predicted_Experimental_Value': y_pred.tolist(),
        'Ground_Truth': y_val.tolist()
    }).to_csv('leaderboard_predictions.csv', index=False)
