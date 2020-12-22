import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data_set = pd.read_csv("dataset", error_bad_lines=False, sep=',', low_memory=False,
                       index_col=0).astype({'Fails35NoPayIn90': 'int32'})
data_set['Fails35NoPayIn90'].hist()
plt.show()