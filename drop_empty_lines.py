import pandas as pd
import numpy as np
df = pd.read_csv("grouped_and_united", comment='#', error_bad_lines=False, sep=',',index_col=0, low_memory=False)
print(df.describe())
df = df.fillna('', inplace=False)
df = df[df['txn_max'] != '']
df = df[df['txn_mean'] != '']
df = df[df['txn_min'] != '']
df = df[df['txn_sum'] != '']
print(df.describe())
df.to_csv("fixed")

