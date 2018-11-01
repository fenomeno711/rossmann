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
stores = stores.drop(columns = ['Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'])
stores['CompetitionOpenSince'] = stores['CompetitionOpenSinceYear'] + ((1/12)*stores['CompetitionOpenSinceMonth']-(1/24))
stores = stores.drop(columns = ['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'])
stores_COS_mean = stores['CompetitionOpenSince'].mean()
stores_CD_mean = stores['CompetitionDistance'].mean()
stores['CompetitionOpenSince'] = stores['CompetitionOpenSince'].fillna(stores_COS_mean)
stores['CompetitionDistance'] = stores['CompetitionDistance'].fillna(stores_CD_mean)

#removing outliers for certain features:
for i in range (0,1115):
    if stores.loc[i, 'CompetitionOpenSince'] < 1999:
        stores.loc[i, 'CompetitionOpenSince'] = 1999

for i in range (0,1115):
    if stores.loc[i, 'CompetitionDistance'] > 30000:
        stores.loc[i, 'CompetitionDistance'] = 30000


rawdata = rawdata.merge(stores, on='Store')

#Transformation of categorial variables
rawdata = pd.get_dummies(rawdata, columns=['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment'])

#getting rid of zero sales data:
rawdata = rawdata[(rawdata['Open'] != 0)]
rawdata = rawdata[(rawdata['Sales'] != 0)]

#Export to csv
rawdata.to_csv('data/preprocessed_data_scaled_ts.csv', index=False)