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
df_train = config.df_train

# Train
def train():
    MODEL(model_file_path = MODEL_FILE_PATH).train(df_train)

train()
