import pandas as pd
import preprocessing as pre

rawdata = pd.read_csv('data/train.csv', low_memory=False)

#data = pre.std_scaler(rawdata.loc['Sales'])


print(rawdata.head())