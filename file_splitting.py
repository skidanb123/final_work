import pandas as pd

print(pd.__version__)
all = pd.read_csv("manualTransactions_funded_all.rpt", delimiter="\t", comment='#', error_bad_lines=False)


def split_dataframe(df, chunk_size=1000000):
    #chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
       # chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
        df[i * chunk_size:(i + 1) * chunk_size].to_csv(path_or_buf="chunks/chunk" + str(i), )
        print(i)
    print("done")
    return 0


split_dataframe(all, 1000000)
