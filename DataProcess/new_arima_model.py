import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


def _get_best_model(timeSeries):
    best_aic = np.inf
    best_order = None
    best_mdl = None
    pq_rng = range(9)
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = ARIMA(timeSeries, order=(i,d,j)).fit(
                        method='css', trend='nc', disp=False
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue

    return best_aic, best_order, best_mdl
