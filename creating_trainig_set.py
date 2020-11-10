import pandas as pd

target_var = pd.read_csv("target.csv", comment='#', error_bad_lines=False, sep=',',
                 low_memory=False,index_col=0).astype({'Fails35NoPayIn90': 'int32'})
another_var = pd.read_csv("fixed", comment='#', error_bad_lines=False, sep=',',
                 low_memory=False,index_col=0)
print(target_var.info())
print("#########################")
print(another_var.info())
final_df = pd.merge(another_var, target_var, left_on='advanceID', right_on='advanceID', how='left')
print(final_df.info())
print("#########################")
final_df = final_df[final_df['Fails35NoPayIn90']!= '']
print(final_df.describe())
print("#########################")
final_df.dropna(how="any",inplace=True)
print(final_df.describe())
final_df.to_csv("dataset")