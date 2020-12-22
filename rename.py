import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from random import randrange

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_seq_items', 12)
pd.set_option('display.max_columns', 12)
all = pd.read_csv("manualTransactions_funded_all.rpt", delimiter="\t", comment='#', error_bad_lines=False)

#print(all.sample(5))
#print("############################")
print(all.info())
print("manualTransactions.rpt")