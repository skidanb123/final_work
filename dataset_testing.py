import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_seq_items', 12)
pd.set_option('display.max_columns', 12)

final_df = pd.read_csv("dataset", comment='#', error_bad_lines=False, sep=',',
                 low_memory=False,index_col=0).astype({'Fails35NoPayIn90': 'int32'})

print(final_df.info())
print("#########################")
print(final_df.describe())
print("#########################")
print(final_df.describe())
#final_df.to_csv("dataset")
print(final_df['Fails35NoPayIn90'].value_counts())
final_df['Fails35NoPayIn90'].hist()
final_df['txn_max'].hist()
final_df['txn_min'].hist()
final_df['txn_mean'].hist()
final_df['txn_sum'].hist()
plt.show()