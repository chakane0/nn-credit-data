import pandas as pd
df_train = pd.DataFrame(pd.read_excel("credit-data.xlsx")) # 999 rows x 21 columns
df_test = pd.DataFrame(pd.read_excel("test-data.xlsx")) # 999 rows x 21 columns


"""
    12  15  31  40  1264  62  73  2  94  101  2.1  122  25  143  151  1  173  1.1  191  201  2.2
0   12  30  34  42  8386  61  74  2  93  101    2  122  49  143  152  1  173    1  191  201    2
1   14  48  32  49  4844  61  71  3  93  101    2  123  33  141  151  1  174    1  192  201    2
2   13  21  32  40  2923  62  73  1  92  101    1  123  28  141  152  1  174    1  192  201    1
"""