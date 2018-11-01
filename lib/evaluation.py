import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, \
    classification_report, matthews_corrcoef, \
    precision_score, recall_score, f1_score, auc, roc_curve, log_loss
import pandas as pd


def save_errors(test, predicted, model, fit_time, pred_time):

    mse = mean_squared_error(test, predicted)
    rmse = np.sqrt(((test - predicted) ** 2).mean())
    r2 = r2_score(test, predicted)
    rmspe = np.sqrt(sum(((test - predicted) / test) ** 2) / len(test))
    rmseom = np.sqrt(((test - predicted) ** 2).mean()) / test.mean()
    cal = predicted.mean() / test.mean()

    results = pd.read_csv('notebooks/results_rossmann2.csv')
    result = pd.DataFrame([[model, fit_time, pred_time, mse, rmse, r2, rmspe, rmseom, cal]],
                  columns=['Model', 'FitTime', 'PredictionTime', 'MSE', 'RMSE', 'R2', 'RMSPE', 'RMSEofMean','Calibration'])
    results = pd.concat([results, result])
    results.to_csv('notebooks/results_rossmann2.csv', index=False)

    print('Results stored in: notebooks/results_rossmann2.csv')

    print(model)
    print('')
    print('Fit finished in: ' + str(fit_time) + 's')
    print('Prediction finished in: ' + str(pred_time) + 's')
    print('')
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('R2: ', r2)
    print('RMSPE: ', rmspe)
    print('RMSE % of mean:', rmseom)
    print('Calibration:', cal)
    print('')