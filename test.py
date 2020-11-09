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


final = pd.DataFrame()
for i in range(149):
    if i != 1:
        chunk0 = pd.read_csv("chunks/chunk" + str(i), comment='#', error_bad_lines=False, sep=',', index_col=0,
                             low_memory=False)
        chunk0 = chunk0.drop(columns=['txnDescription', "txnDate"])
    else:continue
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

    grouped = [grouped_max, grouped_mean, grouped_min, grouped_sum]

    df_final = reduce(lambda left, right: pd.merge(left, right, on='advanceID'), grouped)
    final = final.append(df_final)
    print(i)
final.to_csv("grouped_and_united")

