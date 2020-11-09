import pandas as pd
import numpy as np
from functools import reduce
df = pd.read_csv("grouped_and_united", comment='#', error_bad_lines=False, sep=',', low_memory=False)
print(df.head())
print(df.describe())
#print(df.astype({'advanceID': 'str'}).dtypes)
df = df.drop_duplicates(subset=['advanceID'],keep='last')
print(df.describe())
df.to_csv("grouped_and_united")
#df_final = df.groupby("advanceID").apply()
# print(df_final.describe())
# print(df_final.head())
