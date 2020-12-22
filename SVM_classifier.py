from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

dataset = pd.read_csv("dataset", error_bad_lines=False, sep=',', low_memory=False,
                      index_col=0).astype({'Fails35NoPayIn90': 'int32'})
dataset = dataset.drop("advanceID", axis=1)
msk = np.random.rand(len(dataset)) < 0.90
dataset_less = dataset
print(dataset.info())
print(dataset_less.info())
y = dataset_less['Fails35NoPayIn90']
X = dataset_less.drop('Fails35NoPayIn90', axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.95,
                                                    random_state=42)
svc = OneVsRestClassifier(SVC(kernel='linear',
                              class_weight='balanced', probability=True,tol=1e-2), n_jobs=-1)


print("before fit")
svc.fit(X_train, y_train)
print("after fit")
pred_y_3 = svc.predict(X_test)
pred_pandas = pd.Series(pred_y_3)
print(pred_pandas.value_counts())
print("ACC: " + str(accuracy_score(y_test, pred_y_3)))
prob_y_3 = svc.predict_proba(X_test)
prob_y_3 = [p[1] for p in prob_y_3]
print("roc_auc: " + str(roc_auc_score(y_test, prob_y_3)))
ns_probs = [0 for _ in range(len(y_test))]
print(roc_auc_score(y_test,prob_y_3))
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, prob_y_3)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('SVM: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, prob_y_3)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='SVM_unbalanced')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()