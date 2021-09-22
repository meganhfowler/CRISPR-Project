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


pd.set_option('mode.chained_assignment', None)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# Select Model and File Path
MODEL = models.LinearRegressor2
MODEL_FILE_PATH = "./src/model2.pkl"
HYPERPARAMS_FILE_PATH = "./results/hyperparameters2.csv"

# Get dataframes
df = pd.read_csv("./data/train.csv")
df_train, df_validate = train_test_split(df, test_size = 0.2)
df_test = pd.read_csv("./data/test.csv")

# Train
def train():
    MODEL(model_file_path = MODEL_FILE_PATH).train(df_train)

# Validate
def validate():
    y_predictions, y_actuals = MODEL(model_file_path = MODEL_FILE_PATH).predict(df_validate)
    y_predictions = y_predictions.detach().numpy()
    df_predictions = pd.DataFrame(y_predictions)
    df_predictions.columns = ['prediction']
    y_actuals = y_actuals.detach().numpy()
    return y_predictions, y_actuals

# Test
def test():
    y_predictions, y_actuals = MODEL(model_file_path = MODEL_FILE_PATH).predict(df_test)
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

# Determine Hyperparameters
def benchmark_hyperparams(predictions, actuals):
    mse = mean_squared_error(y_true=actuals, y_pred=predictions)
    correlation, pvalue = spearmanr(a=actuals, b=predictions)
    return mse, correlation, pvalue


def tune_hyperparams(df_train):
    batches = [1, 32, 64, 128, df_train.shape[0]]
    epochs = [1, 3, 5, 7, 9]
    lrs = [0.0001, 0.00001, 0.000001, 0.0000001]
    mse = [0.0]
    correlation = [0.0]
    pvalue = [0.0]
    hyperparams = [batches, epochs, lrs, mse, correlation, pvalue]
    df_hyper = pd.DataFrame(list(itertools.product(*hyperparams)), columns = ["batch", "epoch", "learning_rate", "mse", "correlation", "pvalue"])

    for i in range(len(df_hyper)):
        batch = df_hyper.iloc[i]["batch"]
        epoch = df_hyper.iloc[i]["epoch"]
        lr = df_hyper.iloc[i]["learning_rate"]
        MODEL(model_file_path = MODEL_FILE_PATH).train_hyperparams(df_train, batch, epoch, lr)

        y_predictions, y_actuals = MODEL(model_file_path = MODEL_FILE_PATH).predict(df_validate)
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

    df_hyper.to_csv(HYPERPARAMS_FILE_PATH)

# train()
# preds, actuals = validate()
# print(benchmark(preds, actuals))

preds,actuals = test()
print(benchmark(preds, actuals))


#tune_hyperparams(df_train)
