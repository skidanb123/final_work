import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

dataset = pd.read_csv("dataset", error_bad_lines=False, sep=',', low_memory=False,
                      index_col=0).astype({'Fails35NoPayIn90': 'int32'})
dataset = dataset.drop("advanceID", axis=1)
df_minority = dataset[dataset['Fails35NoPayIn90'] == 1]
df_majority = dataset[dataset['Fails35NoPayIn90'] == 0]

df_minority_upsampled = resample(df_minority, replace=True,
                                 n_samples=430000, random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled['Fails35NoPayIn90'].hist()
plt.show()
print(df_upsampled['Fails35NoPayIn90'].value_counts())
df_upsampled.to_csv("dataset_balanced")
y = df_upsampled['Fails35NoPayIn90']
X = df_upsampled.drop('Fails35NoPayIn90', axis=1)
clf_1 = LogisticRegression().fit(X, y)
pred = clf_1.predict(X)
pred_pandas = pd.Series(pred)
print(pred_pandas.value_counts())
print(accuracy_score(y, pred))
prob_y_2 = clf_1.predict_proba(X)
prob_y_2 = [p[1] for p in prob_y_2]
print("Roc_Auc_Score" +str(roc_auc_score(y,prob_y_2)))
