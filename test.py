import pandas as pd
import numpy as np
from functools import reduce


def mean(x):
    try:
        return np.mean(x)
    except:
        return np.nan


def summ(x):
    try:
        return np.sum(x)
    except:
        return 0


def min(x):
    try:
        return np.min(x)
    except:
        return 0


def max(x):
    try:
        return np.max(x)
    except:
        return 0


chunk0 = pd.read_csv("chunks/chunk0", comment='#', error_bad_lines=False, sep=',', index_col=0, low_memory=False)
chunk0 = chunk0.drop(columns=['txnDescription', "txnDate"])
print(chunk0.head())
for col in chunk0.columns:
    print(col)
##############################
grouped_sum = chunk0.groupby('advanceID').agg(
    {"txnAmount": summ})
grouped_sum = grouped_sum.rename(columns={'txnAmount': 'txn_sum'})
#########################
grouped_mean = chunk0.groupby('advanceID').agg(
    {"txnAmount": mean})
grouped_mean = grouped_mean.rename(columns={'txnAmount': 'txn_mean'})
##############################
grouped_min = chunk0.groupby('advanceID').agg(
    {"txnAmount": min})
grouped_min = grouped_min.rename(columns={'txnAmount': 'txn_min'})
###############################
grouped_max = chunk0.groupby('advanceID').agg(
    {"txnAmount": max})
grouped_max = grouped_max.rename(columns={'txnAmount': 'txn_max'})
###############################
# chunk0=chunk0.groupby('advanceID').aggregate(summ(chunk0.txnAmount))
print("mean ", grouped_mean.head())
print("sum ", grouped_sum.head())
print("min ", grouped_min.head())
print("max ", grouped_max.head())
grouped = [grouped_max, grouped_mean, grouped_min, grouped_sum]
df_final = reduce(lambda left, right: pd.merge(left, right, on='advanceID'), grouped)
print(df_final.head())
