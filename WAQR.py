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
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

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

    def fit_nl(self, tau1, weighting ='es', hidden_sizes=[128, 128], num_epochs=200, loss_function='MSE', batch_size=64,
               use_lr_decay=False, dropout_rate=0.1, lr=0.1, activation='ReLU', step_size=100, gamma=0.1, optimiz='Adam'):
        qnet1 = DQR(self.X, self.Y, options=self.opt)
        if weighting == 'es':
            qnet1.fit(tau=tau1)
        elif weighting == 'sq':
            qnet1.fit(tau=1-tau1)
        qy1 = qnet1.predict(self.X)
        r1 = self.pseudo_outcome(qy1=qy1, tau1=tau1, weighting=weighting)

        if loss_function == 'MSE':
            criterion = nn.MSELoss()
        elif loss_function == 'MAE':
            criterion = nn.L1Loss()
        elif loss_function == 'Huber':
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError("Invalid loss function")

        train_x = torch.tensor(self.X, dtype=torch.float32)
        train_y = torch.tensor(r1, dtype=torch.float32)
        train_data = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        model = FullyConnectedNN(input_size=train_x.shape[1], output_size=1, hidden_sizes=hidden_sizes,
                                 dropout_rate=dropout_rate, activation=activation)
        epoch = 0
        loss_diff = 1
        if optimiz == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=self.opt['weight_decay'],
                                  nesterov=self.opt['nesterov'], momentum=self.opt['momentum'])
        elif optimiz == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=self.opt['weight_decay'])
        else:
            raise Exception(options['optimizer'] + "is currently not available")

        train_losses = []
        if use_lr_decay:
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        while epoch < num_epochs and loss_diff > 1e-4:
            train_loss = torch.Tensor([0])
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y.view_as(predictions))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if np.isnan(loss.data.numpy()):
                    import warnings
                    warnings.warn('NaNs encountered in training model.')
                    break

            train_losses.append(train_loss / len(train_loader))
            if epoch != 0:
                loss_diff = np.abs(train_losses[epoch]-train_losses[epoch-1])
            if use_lr_decay:
                scheduler.step()
            epoch += 1

        return model, train_losses



class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout_rate, activation='ReLU'):
        super(FullyConnectedNN, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        if activation == "ReLU":
            self.activation = nn.ReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid
        else:
            raise Exception(activation + "is currently not available as a activation function")

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(self.activation())
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, X):
        tX = torch.tensor(X, dtype=torch.float32)
        self.eval()
        self.zero_grad()
        yhat = self.forward(tX)[:, 0]
        return yhat.data.numpy()

















