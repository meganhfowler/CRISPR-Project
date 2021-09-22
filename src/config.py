#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import models
from sklearn.model_selection import train_test_split


# Select Model and File Path
MODEL = models.LinearRegressor2
MODEL_FILE_PATH = "./src/model2.pkl"
HYPERPARAMS_FILE_PATH = "./results/hyperparameters/model2.csv"
PREDICTIONS_FILE_PATH = "./results/predictions/model2.csv"

# Get dataframes
df = pd.read_csv("./data/train.csv")
df_train, df_validate = train_test_split(df, test_size = 0.2)
df_test = pd.read_csv("./data/test.csv")

# Predict validation or test set?
df_predict = df_validate


