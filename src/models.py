#!/usr/bin/env python
# coding: utf-8

import torch
from transformations import Preprocessing
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import pickle

class LinearRegressor:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.params = ["grna_target_sequence", "target_sequence"]


    def train(self, df_train):
        df_train = Preprocessing.drop_na(df_train, self.params)
        X = Preprocessing.get_X(df_train, self.params)
        y = Preprocessing.get_y(df_train)

        input_dim = X.shape[1]
        X = Variable(X)
        y = Variable(y)
        model = nn.Linear(input_dim, 1)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)
        BATCH_SIZE = 20
        EPOCH = 1
        torch_dataset = Data.TensorDataset(X, y)
        loader = Data.DataLoader(
            dataset = torch_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 2
        )
        for epoch in range(EPOCH):
            for step, (batch_x, batch_y) in enumerate(loader):
                prediction = model(batch_x)
                loss = loss_func(prediction, batch_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)

    def predict(self, df_test):
        with open(self.model_file_path, 'rb') as model_file:
            model: model = pickle.load(model_file)

        df_test = Preprocessing.drop_na(df_test, self.params)
        X = Preprocessing.get_X(df_test, self.params)
        y_predictions = model(X)
        y_actuals = Preprocessing.get_y(df_test)
        return y_predictions, y_actuals




