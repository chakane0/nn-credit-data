import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
#sklearn 
import sklearn
from sklearn.cluster import KMeans 
from sklearn.preprocessing import scale # for scaling the data
import sklearn.metrics as sm # for evaluating the model
from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report


rcParams["figure.figsize"] = 20, 10

df_train = pd.DataFrame(pd.read_excel("credit-data.xlsx")) # 999 rows x 21 columns -> (999, 21)


data = scale(df_train)
print(data)
