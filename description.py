import pandas as pd
chunk0 = pd.read_csv("chunks/chunk0",delimiter="\t", comment='#', error_bad_lines=False)
print(chunk0.head)