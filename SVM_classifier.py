from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("dataset_balanced", error_bad_lines=False, sep=',', low_memory=False,
                      index_col=0).astype({'Fails35NoPayIn90': 'int32'})
msk = np.random.rand(len(dataset)) < 0.90
dataset_less = dataset[~msk]
print(dataset.info())
print(dataset_less.info())
y = dataset_less['Fails35NoPayIn90']
X = dataset_less.drop('Fails35NoPayIn90', axis=1)
svc = OneVsRestClassifier(SVC(kernel='linear',
                              class_weight='balanced', probability=True,tol=1e-4), n_jobs=-1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print("before fit")
svc.fit(X, y)
print("after fit")
pred_y_3 = svc.predict(X)
print("ACC: " + str(accuracy_score(y, pred_y_3)))
prob_y_3 = svc.predict_proba(X)
prob_y_3 = [p[1] for p in prob_y_3]
print("roc_auc: " + str(roc_auc_score(y, prob_y_3)))
