from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import time
import evaluation


def k_neighbor_regressor(X_train, X_test, y_train, y_test, *, n_neighbors=5,
                         weights='uniform', algorithm='auto', leaf_size=30,
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



def linear_regression(X_train, X_test, y_train, y_test, *, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
    linReg = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)

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
                  bootstrap=True, oob_score=False, n_jobs=1,
                  random_state=None, verbose=0, warm_start=False):

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
