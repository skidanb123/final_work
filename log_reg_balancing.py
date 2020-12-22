import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.utils import resample

data_set = pd.read_csv("dataset", error_bad_lines=False, sep=',', low_memory=False,
                       index_col=0).astype({'Fails35NoPayIn90': 'int32'})
data_set = data_set.drop("advanceID", axis=1)
y = data_set['Fails35NoPayIn90']
X = data_set.drop('Fails35NoPayIn90', axis=1)
df_minority = data_set[data_set['Fails35NoPayIn90'] == 1]
df_majority = data_set[data_set['Fails35NoPayIn90'] == 0]

df_minority_upsampled = resample(df_minority, replace=True,
                                 n_samples=430000, random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df_upsampled['Fails35NoPayIn90'].hist()
y = df_upsampled['Fails35NoPayIn90']
X = df_upsampled.drop('Fails35NoPayIn90', axis=1)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.15,
                                                    random_state=42)
clf_0 = LogisticRegression().fit(X_train, y_train)
pred = clf_0.predict(X_test)
print("acc:" +str(accuracy_score(pred, y_test)))
pred_pandas = pd.Series(pred)
print(pred_pandas.value_counts())
prob_y_2 = clf_0.predict_proba(X_test)
prob_y_2 = [p[1] for p in prob_y_2]
ns_probs = [0 for _ in range(len(y_test))]
print(roc_auc_score(y_test,prob_y_2))
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, prob_y_2)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, prob_y_2)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic_Balanced')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()