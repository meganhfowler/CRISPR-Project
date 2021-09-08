#!/usr/bin/env python3
import numpy as np
import pandas as pd
import models
import transformations
import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

# Load data set
df_train = pd.read_csv("data/df_train.csv")
df_train_X = df_train.drop(columns = ["cleavage_freq"])
df_train_y = df_train["cleavage_freq"]

# Preprocessing
class Preprocessing:
    def fold_seq(df):
        df["stacked"] = df["grna_target_sequence"].apply(lambda x: x.tolist()) + df["target_sequence"].apply(lambda x: x.tolist())
        df["stacked"] = df["stacked"].apply(lambda x: np.array(x))

    def tensorfy(stacked):
        temp = []
        for i in stacked:
            temp.append(i)
        return torch.from_numpy(np.array(temp).astype(np.float32))
params = ["grna_target_sequence", "target_sequence"]
X = transformations.Preprocessing.preprocess(df_train_X, params)
y = torch.Tensor(np.array(df_train_y).reshape(df_train_y.shape[0], 1).astype(np.float32))
print(y.shape)
train_ds = TensorDataset(X, y)

# Train
models.LinearRegressor('code/model.pickle', train_ds, X, y).train()

# Analysis
y_predictions = models.LinearRegressor('code/model.pickle', train_ds, X, y).predict(X)
def benchmark(predictions, actuals):
    predictions = predictions.detach().numpy()
    actuals = actuals.detach().numpy()
    mse = mean_squared_error(y_true=actuals, y_pred=predictions)
    correlation, pvalue = spearmanr(a=actuals, b=predictions)

    return {
        'mean_squared_error': mse,
        'spearman_rank': {
            'correlation':  correlation,
            'pvalue': pvalue,
        }
    }
print(benchmark(y_predictions, y))


