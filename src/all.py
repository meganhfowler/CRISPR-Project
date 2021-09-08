#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import models
import torch
from transformations import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


if torch.cuda.is_available():
    dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

df = pd.read_csv("./data/train.csv")
df_train, df_validate = train_test_split(df, test_size = 0.2)

# Train
models.LinearRegressor(model_file_path = "./src/model.pkl").train(df_train)




# Validate

y_predictions, y_actuals = models.LinearRegressor(model_file_path = "./src/model.pkl").predict(df_validate)
y_predictions = y_predictions.detach().numpy()
df_predictions = pd.DataFrame(y_predictions)
df_predictions.columns = ['prediction']
y_actuals = y_actuals.detach().numpy()




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

print(benchmark(y_predictions, y_actuals))
