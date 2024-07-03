import sys
sys.path.append('..')

from dataloader import DataLoader
from xgboost import XGBClassifier
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import roc_auc_score, auc

data = DataLoader()
data.load_benchmark("mayhew_2022")
data.featurize("rdkit2d_normalized_features")

X = data.features
y = data.labels

X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)

bst = XGBClassifier(n_estimators=10000, max_depth=10, learning_rate=0.1, min_child_weight=1, subsample=1, colsample_bytree=1,
                        early_stopping_rounds=20, eval_metric='auc', n_jobs=-1, random_state=42)

bst.fit(X_train, y_train,eval_set=[(X_test, y_test)])
y_pred = bst.predict(X_test)

print(roc_auc_score(y_pred, y_test))
