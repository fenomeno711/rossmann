import pandas as pd
import lib
import xgboost as xgb
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/preprocessed_data.csv', sep=',')
X = data.drop(columns=['Customers'])
X = X.dropna(axis=0)
y = X.loc[:,'Sales']
X = X.drop(columns=['Sales'])


#splitting the data set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#splitting the data set by timestamp

data = data.drop(columns=['Customers'])
data = data.dropna(axis=0)

timestamp = data[(data['Day'] == 15) & (data['Month'] == 6) & (data['Year'] == 2015)]['Timestamp'].values[0]
testdata = data[data['Timestamp'] >= timestamp]
traindata = data[data['Timestamp'] < timestamp]

print('Number of samples trainings:', len(traindata))
print('Number of samples test:', len(testdata))


y_train = traindata.loc[:,'Sales']
X_train = traindata.drop(columns=['Sales'])

y_test = testdata.loc[:,'Sales']
X_test = testdata.drop(columns=['Sales'])


#lib.k_neighbor_regressor(X_train, X_test, y_train, y_test)

#lib.linear_regression(X_train, X_test, y_train, y_test)

#lib.decision_tree(X_train, X_test, y_train, y_test)

lib.random_forest(X_train, X_test, y_train, y_test)

# xgtrain = xgb.DMatrix(X_train.values, y_train.values)
# xgtest = xgb.DMatrix(X_test.values)

#lib.XGBoost(X_train, X_test, y_train, y_test)

#lib.GBR(X_train, X_test, y_train, y_test)