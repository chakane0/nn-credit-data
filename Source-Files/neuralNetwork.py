import pandas as pd
import numpy as np
df_train = pd.DataFrame(pd.read_excel("credit-data.xlsx")) # 999 rows x 21 columns -> (999, 21)
df_test = pd.DataFrame(pd.read_excel("test-data.xlsx")) # 20 rows x 21 columns -> (20, 21)

""" 
convert our dataframe into a matrix which represents our vector inputs
i.e)

[[ 12  48  32 ... 191 201   2]
 [ 14  12  34 ... 191 201   1]
 [ 11  42  32 ... 191 201   1]
 ...
 [ 14  12  32 ... 191 201   1]
 [ 11  45  32 ... 192 201   2]
 [ 12  45  34 ... 191 201   1]]

"""
training_input_matrix = df_train.to_numpy().T
training_output_matrix = df_test.to_numpy().T



print(training_input_matrix)