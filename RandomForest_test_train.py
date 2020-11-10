import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data_set = pd.read_csv("dataset_balanced", error_bad_lines=False, sep=',', low_memory=False,
                       index_col=0).astype({'Fails35NoPayIn90': 'int32'})
y = data_set['Fails35NoPayIn90']
X = data_set.drop('Fails35NoPayIn90', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.85,
                                                    random_state=42)
clf_2 = RandomForestClassifier().fit(X_train, y_train)
pred_y_2 = clf_2.predict(X_test)
print(accuracy_score(y_test, pred_y_2))  # 95% 0.7132882492623605 90% 0.8357930585001558

pred_pandas = pd.Series(pred_y_2)
print(pred_pandas.value_counts())
# 0    432678
# 1     24460
prob_y_2 = clf_2.predict_proba(X_test)
prob_y_2 = [p[1] for p in prob_y_2]
print(roc_auc_score(y_test, prob_y_2))  # 95%  0.801782441307876 90% 0.9161910577984638