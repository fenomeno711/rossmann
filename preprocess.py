import pandas as pd
import preprocessing as pre

rawdata = pd.read_csv('data/train.csv', low_memory=False)

#Transformation of Date
rawdata['Day'] = rawdata['Date'].apply(pre.convert2day)
rawdata['Month'] = rawdata['Date'].apply(pre.convert2month)
rawdata['Year'] = rawdata['Date'].apply(pre.convert2year)
rawdata['Timestamp'] = rawdata['Date'].apply(pre.convert2timestamp)

rawdata = rawdata.drop(columns='Date')

#Merging with store_data
stores = pd.read_csv('data/store.csv')

#...insert preprocessing for stores here (missing values etc.)

rawdata = rawdata.merge(stores, on='Store')

#Transformation of categorial variables
rawdata = pd.get_dummies(rawdata, columns=['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment'])

#Export to csv
rawdata.to_csv('data/preprocessed_data.csv')