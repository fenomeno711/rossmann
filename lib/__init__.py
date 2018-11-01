from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
#import xgboost as xgb
import time
import evaluation
import pandas as pd


def k_neighbor_regressor(X_train, X_test, y_train, y_test, *, n_neighbors=3,
                         weights='uniform', algorithm='auto', leaf_size=3,
                         p=2, metric='minkowski', metric_params=None):
    knr = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights,
                              algorithm=algorithm, leaf_size=leaf_size,
                              p=p, metric=metric, metric_params=metric_params)
    model = knr
    fit_start = time.time()
    knr.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = knr.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, y_prediction, model, fit_time, pred_time)

    return y_test, y_prediction, model, fit_time, pred_time



def linear_regression(X_train, X_test, y_train, y_test, *, normalize=False, copy_X=True, n_jobs=None):
    linReg = LinearRegression(normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)

    model = linReg
    fit_start = time.time()
    linReg.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = linReg.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, y_prediction, model, fit_time, pred_time)
    w = linReg.coef_
    print(w)
    export = pd.DataFrame(columns = ['y_test', 'y_prediction'])
    export['y_test'] = y_test
    export['y_prediction'] = y_prediction
    export.to_csv('notebooks/export.csv', index=False)


""" --------------------------------------------------------------------
decision_tree() takes input: X_train, X_test, y_train, y_test
fits DecisionTreeRegressor, computes y_prediction,
writes results to 'result' and prints 'errors'
-------------------------------------------------------------------- """









def decision_tree(X_train, X_test, y_train, y_test, *, max_depth=None,
                  random_state=None):

    dTree = DecisionTreeRegressor(max_depth=max_depth,
                                  random_state=random_state)

    model = str(dTree)

    fit_start = time.time()
    dTree.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = dTree.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, y_prediction, model, fit_time, pred_time)








""" --------------------------------------------------------------------
random_forest() takes input: X_train, X_test, y_train, y_test
fits RandomForestRegressor and returns as output: y_test, y_prediction
-------------------------------------------------------------------- """


def random_forest(X_train, X_test, y_train, y_test, *, n_estimators=10,
                  criterion='mse', max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                  max_features='auto', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  bootstrap=True, oob_score=False, n_jobs=2,
                  random_state=None, verbose=1, warm_start=False):

    regr = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion, max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split,
        bootstrap=bootstrap, oob_score=oob_score,
        n_jobs=n_jobs, random_state=random_state,
        verbose=verbose, warm_start=warm_start)

    model = str(regr)

    fit_start = time.time()
    regr.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = regr.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, y_prediction, model, fit_time, pred_time)

    export = pd.DataFrame(columns=['y_test', 'y_prediction'])
    export['y_test'] = y_test
    export['y_prediction'] = y_prediction
    export.to_csv('notebooks/export_rf.csv', index=False)


def XGBoost(X_train, X_test, y_train, y_test, *, objective='reg:linear', colsample_bytree = 0.3, learning_rate=0.1,max_depth=5, alpha=10, n_estimators=10):

    xg_reg = xgb.XGBRegressor(objective=objective, colsample_bytree=colsample_bytree, learning_rate=learning_rate,
                              max_depth=max_depth, alpha=alpha, n_estimators=n_estimators)

    model = str(xg_reg)

    fit_start = time.time()
    xg_reg.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = xg_reg.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, y_prediction, model, fit_time, pred_time)


def GBR(X_train, X_test, y_train, y_test, *,loss='ls', learning_rate=0.2, n_estimators=30, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.8, verbose=1, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001):


    GBR = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, init=init, random_state=random_state, max_features=max_features, alpha=alpha, verbose=verbose    , max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, presort=presort, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol)

    model = str(GBR)

    fit_start = time.time()
    GBR.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = GBR.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, y_prediction, model, fit_time, pred_time)
