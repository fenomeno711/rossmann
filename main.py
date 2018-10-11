import pandas as pd
import lib
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/preprocessed_data.csv', sep=',')
X = data.drop(columns=['Customers', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'])
X = X.dropna(axis=0)
y = X.loc[:,'Sales']
X = X.drop(columns=['Sales'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#lib.k_neighbor_regressor(X_train, X_test, y_train, y_test)

lib.linear_regression(X_train, X_test, y_train, y_test)

lib.random_forest(X_train, X_test, y_train, y_test)