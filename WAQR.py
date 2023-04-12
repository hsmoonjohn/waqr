import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import os
# import sys
# import torch
# import matplotlib.pyplot as plt
# import time
from DeepQuantile import DQR
from sklearn.linear_model import LinearRegression



class WAQR:
    '''
    Weighted-Average Quantile Regression
    '''
    opt = {'momentum': 0.9, 'nesterov': True, 'lr_decay': 0.9,
           'dropout_proportion': 0, 'Lambda': 0,
           'lr': 1e-1, 'weight_decay': 0.0, 'tol': 1e-4,
           'width': 200, 'depth': 3}

    def __init__(self, X, Y, standardize=False, options=dict()):
        '''
        Arguments
        ---------
        X : n by p matrix of covariates; each row is an observation vector.

        Y : an ndarray of response variables.

        standardize : logical flag for x variable standardization prior to fitting the model;
                      default is FALSE.
                      default is FALSE.

        options : a dictionary of neural network and optimization parameters.
        ---------

        '''
        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.stand = standardize
        self.X = X
        # self.X_dqr, self.X_ls, self.Y_dqr, self.Y_ls = train_test_split(self.X, self.Y, test_size=QL_ratio)
        self.opt.update(options)

    def pseudo_outcome(self, qy1, tau1, weighting = 'es'):

        'Calculates pseudo outcome that used for the regression'
        # Expected Shortfall
        if weighting == 'es':
            return qy1+((self.Y-qy1)/tau1)*np.array(self.Y <= qy1)
        # Superquantile
        elif weighting == 'sq':
            return qy1+((self.Y-qy1)/tau1)*np.array(self.Y >= qy1)

        else:
            raise Exception(weighting + "is currently not available as a weight")

    def fit_ls(self, tau1,  weighting = 'es', intercept=False):
        qnet1 = DQR(self.X, self.Y, options=self.opt)
        if weighting == 'es':
            qnet1.fit(tau=tau1)
        elif weighting == 'sq':
            qnet1.fit(tau=1-tau1)
        qy1 = qnet1.predict(self.X)
        r1 = self.pseudo_outcome(qy1=qy1, tau1=tau1, weighting=weighting)
        return LinearRegression(fit_intercept=intercept).fit(self.X, r1)

    def fit_nl(self, tau1, weighting ='es', hidden_sizes=[128,128], num_epochs=200, loss_function = 'MSE'):
        qnet1 = DQR(self.X, self.Y, options=self.opt)
        if weighting == 'es':
            qnet1.fit(tau=tau1)
        elif weighting == 'sq':
            qnet1.fit(tau=1-tau1)
        qy1 = qnet1.predict(self.X)
        r1 = self.pseudo_outcome(qy1=qy1, tau1=tau1, weighting=weighting)
        model=FullyConnectedNN(input_size=self.X.shape[1], output_size=1, hidden_sizes=hidden_sizes)
        if loss_function == 'MSE':
            criterion = nn.MSELoss()
        elif loss_function == 'MAE':
            criterion = nn.L1Loss()
        elif loss_function == 'Huber':
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError("Invalid loss function")

        optimizer = optim.Adam(model.parameters(),lr=self.opt['lr'])
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(train_x)
            loss = criterion(predictions, train_y)
            loss.backward()
            optimizer.step()





class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(FullyConnectedNN, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x















