#!/usr/bin/env python3
import pandas as pd
import models
import torch

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

# Load data set
df_train = pd.read_csv("data/train.csv")

# Train
models.LinearRegressor(model_file_path = 'code/model.pickle').train(df_train)

