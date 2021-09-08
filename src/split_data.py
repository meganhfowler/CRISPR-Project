#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/raw_data.csv")
df_train, df_test = train_test_split(df, test_size = 0.2)
df_train.to_csv("./data/train.csv")
df_test.to_csv("./data/test.csv")



