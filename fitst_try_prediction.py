import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

data_set = pd.read_csv("dataset", error_bad_lines=False, sep=',', low_memory=False,
                       index_col=0).astype({'Fails35NoPayIn90': 'int32'})
data_set = data_set.drop("advanceID", axis=1)
y = data_set['Fails35NoPayIn90']
X = data_set.drop('Fails35NoPayIn90', axis=1)
clf_0 = LogisticRegression().fit(X, y)
pred = clf_0.predict(X)
print(accuracy_score(pred, y))
pred_pandas = pd.Series(pred)
print(pred_pandas.value_counts())
