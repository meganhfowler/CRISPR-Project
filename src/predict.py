#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import models
import torch
from transformations import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import itertools
import config

pd.set_option('mode.chained_assignment', None)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# Get Model, File Path, DataFrames from config
MODEL = config.MODEL
MODEL_FILE_PATH = config.MODEL_FILE_PATH
PREDICTIONS_FILE_PATH = config.PREDICTIONS_FILE_PATH
df_predict = config.df_predict


# Predict
def predict(df_predict):
    y_predictions, y_actuals, df_predict = MODEL(model_file_path = MODEL_FILE_PATH).predict(df_predict)
    y_predictions = y_predictions.detach().numpy()
    df_predictions = pd.DataFrame(y_predictions)
    df_predictions.columns = ["predictions"]
    y_actuals = y_actuals.detach().numpy()
    df_predict["predictions"] = y_predictions
    df_predict.to_csv(PREDICTIONS_FILE_PATH)
    return y_predictions, y_actuals

# Benchmark
def benchmark(predictions, actuals):
    mse = mean_squared_error(y_true=actuals, y_pred=predictions)
    correlation, pvalue = spearmanr(a=actuals, b=predictions)
    return {
            'mean_squared_error': mse,
            'spearman_rank': {
                'correlation':  correlation,
                'pvalue': pvalue,
                }
            }

preds,actuals = predict(df_predict)
print(benchmark(preds, actuals))



