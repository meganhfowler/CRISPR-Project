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


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

df = pd.read_csv("./data/train.csv")
df_train, df_validate = train_test_split(df, test_size = 0.2)
df_test = pd.read_csv("./data/test.csv")

# Train
def train():
    models.LinearRegressor(model_file_path = "./src/model.pkl").train(df_train)


# Validate
def validate():
    y_predictions, y_actuals = models.LinearRegressor(model_file_path = "./src/model.pkl").predict(df_validate)
    y_predictions = y_predictions.detach().numpy()
    df_predictions = pd.DataFrame(y_predictions)
    df_predictions.columns = ['prediction']
    y_actuals = y_actuals.detach().numpy()
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

# Test
def test():
    y_predictions, y_actuals = models.LinearRegressor(model_file_path = "./src/model.pkl").predict(df_test)
    y_predictions = y_predictions.detach().numpy()
    df_predictions = pd.DataFrame(y_predictions)
    df_predictions.columns = ['prediction']
    y_actuals = y_actuals.detach().numpy()
    return y_predictions, y_actuals




# Hyperparameters
def benchmark_hyperparams(predictions, actuals):
    mse = mean_squared_error(y_true=actuals, y_pred=predictions)
    correlation, pvalue = spearmanr(a=actuals, b=predictions)
    return mse, correlation, pvalue


def tune_hyperparams(df_train):
    batches = [1, 32, 64, 128, df_train.shape[0]]
    epochs = [1, 3, 5, 7, 9]
    lrs = [0.1, 0.01, 0.001, 0.0001]
    mse = [0.0]
    correlation = [0.0]
    pvalue = [0.0]
    hyperparams = [batches, epochs, lrs, mse, correlation, pvalue]
    df_hyper = pd.DataFrame(list(itertools.product(*hyperparams)), columns = ["batch", "epoch", "learning_rate", "mse", "correlation", "pvalue"])


    for i in range(len(df_hyper)):
        batch = df_hyper.iloc[i]["batch"]
        epoch = df_hyper.iloc[i]["epoch"]
        lr = df_hyper.iloc[i]["learning_rate"]
        models.LinearRegressor(model_file_path = "./src/model.pkl").train_hyperparams(df_train, batch, epoch, lr)

        # Validate

        y_predictions, y_actuals = models.LinearRegressor(model_file_path = "./src/model.pkl").predict(df_validate)
        y_predictions = y_predictions.detach().numpy()
        df_predictions = pd.DataFrame(y_predictions)
        df_predictions.columns = ['prediction']
        y_actuals = y_actuals.detach().numpy()
        try:
            mse, correlation, pvalue = benchmark_hyperparams(y_predictions, y_actuals)
        except:
            mse = float("NaN")
            correlation = float("NaN")
            pvalue = float("NaN")
        df_hyper.at[i, "mse"] = mse
        df_hyper.at[i, "correlation"] = correlation
        df_hyper.at[i, "pvalue"] = pvalue

    df_hyper.to_csv("./results/hyperparameters.csv")


preds, actuals = test()
print(benchmark(preds, actuals))
