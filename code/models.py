#!/usr/bin/env python3
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Model 1
class LinearRegressor:
    def __init__(self, model_file_path, train_ds, X, y):
        self.model_file_path = model_file_path
        self.batch_size = 5
        self.train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)
        self.model = nn.Linear(100, 1)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=1e-5)
        self.loss_fn = F.mse_loss
        self.loss = self.loss_fn(self.model(X), y)
        self.num_epochs = 100

    def train(self):
        for epoch in range(self.num_epochs):
            for xb,yb in self.train_dl:
                # Generate predictions
                pred = self.model(xb)
                # print(pred)
                loss = self.loss_fn(pred, yb)
                # Perform gradient descent
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

        with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)


    def predict(self, X):
        with open(self.model_file_path, 'rb') as model_file:
            model: model = pickle.load(model_file)
        return model(X)

