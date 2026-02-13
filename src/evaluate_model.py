from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

from src.config import *

#  Function to evaluate the model
def evaluate_model(X, true, predicted):
    n = len(true)
    p = X.shape[1]
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    adj_rsquare = 1 - (1 - r2) * (n - 1) / (n - p - 1)    
    return mae, rmse, r2, adj_rsquare



