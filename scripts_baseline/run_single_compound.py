import sys
sys.path.append('..')

import json

from dataloader import DataLoader
from xgboost import XGBClassifier, XGBRegressor
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau
import numpy as np
import pandas as pd
import sys

def get_xgboost_model(task):
    model_map = {
        'binary': XGBClassifier(tree_method="hist", n_estimators=200, max_depth=10, learning_rate=0.1,
                early_stopping_rounds=10, eval_metric='logloss', n_jobs=-1, verbosity=0, random_state=42),
        'regression': XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=10, eval_metric=mean_squared_error, n_jobs=-1, random_state=42, verbosity=0),
    }
    return model_map[task]

if __name__ == '__main__':
    dataset_names = DataLoader().get_dataset_names()
    results = {k: [] for k in dataset_names}

    for name in dataset_names:
        print(f'Working on {name}...')
        data = DataLoader()
        try:
            data.load_benchmark(name)
        except:
            print(f'Error in {name} loading')
            continue
        data.featurize("rdkit2d_normalized_features")
        spec = DataLoader().get_dataset_specifications(name)

        # setup train test split
        X = data.features
        y = data.labels
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)

        # setup model
        task = spec['task']
        task_dim = spec['task_dim']
        model = get_xgboost_model(task)

        # train the model with early stopping
        stats = []
        if task == 'binary':
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            y_pred = model.predict_proba(X_test)
            if task_dim == 1:
                y_pred = y_pred[:, 1]
            try:
                stats.append({'auc': roc_auc_score(y_test, y_pred)})
            except:
                print(f'Error in {name} calculating AUC metric')
                continue
        else:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            y_pred = model.predict(X_test)
            stats.append({'r2': r2_score(y_test, y_pred),
                        'prs': pearsonr(y_test.squeeze(), y_pred.squeeze())[0]})

        results[name] = pd.DataFrame(stats).mean().to_dict()

    json.dump(results, open('single_compound_results.json', 'w'), indent=4)


