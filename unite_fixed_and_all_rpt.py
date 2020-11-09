import pandas as pd
import matplotlib.pyplot as plt

###################

# all = pd.read_csv("all.rpt", delimiter="\t", comment='#', error_bad_lines=False, warn_bad_lines=True)
# all.dropna(inplace=True)
# all = all.astype({'Fails35NoPayIn90': 'int32'})
# all.to_csv("all_saved.csv")

###################

df = pd.read_csv("all_saved.csv", comment='#', error_bad_lines=False, sep=',',
                 low_memory=False,index_col=0).astype({'Fails35NoPayIn90': 'int32'})
df = df[['advanceID','Fails35NoPayIn90']]
print(df.head())
df.hist()
plt.show()
print(df['Fails35NoPayIn90'].value_counts())

##########
df.to_csv("targer.csv")