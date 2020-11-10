import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Trying RandomForest on Balance Set
data_set = pd.read_csv("dataset_balanced", error_bad_lines=False, sep=',', low_memory=False,
                       index_col=0).astype({'Fails35NoPayIn90': 'int32'})
y = data_set['Fails35NoPayIn90']
X = data_set.drop('Fails35NoPayIn90', axis=1)
clf_2 = RandomForestClassifier().fit(X, y)
pred_y_2 = clf_2.predict(X)
print(accuracy_score(y, pred_y_2))  # 0.9999999987502278

pred_pandas = pd.Series(pred_y_2)
print(pred_pandas.value_counts())
# 0    432678
# 1     24460
prob_y_2 = clf_2.predict_proba(X)
prob_y_2 = [p[1] for p in prob_y_2]
print(roc_auc_score(y, prob_y_2))  # 0.9999999987502278
# Okay, it`s overfit
