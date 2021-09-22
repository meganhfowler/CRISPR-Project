 #!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data
import pickle

from transformations import Preprocessing

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
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
        BATCH_SIZE = 1
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

    def predict(self, df_predict):
        with open(self.model_file_path, 'rb') as model_file:
            model: model = pickle.load(model_file)

        df_predict = Preprocessing.drop_na(df_predict, self.params)
        X = Preprocessing.get_X(df_predict, self.params)
        y_predictions = model(X)
        y_actuals = Preprocessing.get_y(df_predict)
        return y_predictions, y_actuals, df_predict

    def train_hyperparams(self, df_train, batch, epoch, learning_rate):
        df_train = Preprocessing.drop_na(df_train, self.params)
        X = Preprocessing.get_X(df_train, self.params)
        y = Preprocessing.get_y(df_train)

        input_dim = X.shape[1]
        X = Variable(X)
        y = Variable(y)
        model = nn.Linear(input_dim, 1)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        BATCH_SIZE = int(batch)
        EPOCH = int(epoch)
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



class LinearRegressor2:

    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.params = ["target_strand",
          "grna_target_strand",
          "target_sequence",
          "grna_target_sequence",
          "energy_1",
          "energy_2",
          "energy_3",
          "energy_4",
          "energy_5",
          "study_name",
          "whole_genome",
          "delivery_mode"
         ]

    def train(self, df_train):
        df_train = Preprocessing.drop_na(df_train, self.params)
        X = Preprocessing.get_X_2(df_train, self.params)
        y = Preprocessing.get_y(df_train)

        input_dim = X.shape[1]
        X = Variable(X)
        y = Variable(y)

        model = nn.Linear(input_dim, 1)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
        BATCH_SIZE = 1
        EPOCH = 9

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

    def predict(self, df_predict):
        with open(self.model_file_path, 'rb') as model_file:
            model: model = pickle.load(model_file)

        df_predict = Preprocessing.drop_na(df_predict, self.params)
        X = Preprocessing.get_X_2(df_predict, self.params)
        y_predictions = model(X)
        y_actuals = Preprocessing.get_y(df_predict)
        return y_predictions, y_actuals, df_predict

    def train_hyperparams(self, df_train, batch, epoch, learning_rate):
        df_train = Preprocessing.drop_na(df_train, self.params)
        X = Preprocessing.get_X_2(df_train, self.params)
        y = Preprocessing.get_y(df_train)

        input_dim = X.shape[1]
        X = Variable(X)
        y = Variable(y)
        model = nn.Linear(input_dim, 1)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        BATCH_SIZE = int(batch)
        EPOCH = int(epoch)
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



