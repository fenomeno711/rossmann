import pandas as pd
import preprocessing as pre

rawdata = pd.read_csv('data/train.csv', low_memory=False)

#Transformation of Date
rawdata['Day'] = rawdata['Date'].apply(pre.convert2day)
rawdata['Month'] = rawdata['Date'].apply(pre.convert2month)
rawdata['Year'] = rawdata['Date'].apply(pre.convert2year)
rawdata['Timestamp'] = rawdata['Date'].apply(pre.convert2timestamp)

rawdata.drop(columns='Date')

#Transformation of Weekday
weekdays = pd.get_dummies(rawdata['DayOfWeek'])
rawdata.join(weekdays)

rawdata.drop(columns = ['DayOfWeek'])

#Transformation of StateHoliday


#Merging with store_data
stores = pd_read_csv('data/store.csv')

#...insert preprocessing for stores here (missing values etc.)

rawdata.merge(stores, on='Store')

#Export to csv
rawdata.to_csv('data/preprocessed_data.csv')