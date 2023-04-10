from distutils.command.config import config
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time



class DQR:
    '''
    Deep Quantile Regression
    '''

    optimizers = ["SGD", "Adam"]
    activations = ["sigmoid", "tanh", "ReLU"]
    opt = {'momentum' : 0.9, 'nesterov' : True, 'lr_decay' : 0.9, 
           'dropout_proportion' : 0, 'Lambda' : 0, 
           'lr' : 1e-1, 'weight_decay' : 0.0, 'tol' : 1e-4,
           'width' : 200, 'depth' : 3}
    
    def __init__(self, X, Y, standardize = False, plot = False, options = dict()):
        '''
        Arguments
        ---------
        X : n by p matrix of covariates; each row is an observation vector.
           
        Y : an ndarray of response variables.

        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is FALSE.
                      
        options : a dictionary of neural network and optimization parameters.
        ---------
            momentum : momentum accerleration rate for SGD algorithm; default is 0.9.
            nesterov : logical flag for using Nesterov gradient descent algorithm; default is TRUE.
            lr_decay : multiplicative factor of learning rate decay; default is 0.9.
            dropout_proportion : proportion of the dropout; default is 0.
            Lambda : regularization parameter for L_1 regularization; default is 0.
            lr : learning rate of SGD or Adam optimization; default is 1e-1.
            weight_decay : weight decay of L2 penalty; default is 0.
            tol : the iteration will stop when the difference between consecutive losses 
                  is less then tol; default is 1e-4.
            width : the number of neurons for each layer; default is 200.
            depth : the number of hidden layers; default is 3.
        '''
        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.stand = standardize
        self.opt.update(options)
        self.X = X
        self.plot = plot

    def fit(self, tau, nepochs = 100, optimizer = 'SGD', activation = 'ReLU',
            batch_size = 64):
        '''
        Arguments
        ---------
        tau : quantile level between 0 and 1; default is 0.5.
        nepochs : number of epochs; default is 100.
        optimizer : default is the Adam optimizer.
        activation : defulat is the ReLU function
        batch_size : number of data in one batch; default is 64.
        '''
        
        self.model = fit_quantile(self.X, self.Y, tau = tau, nepochs=200, 
                                  optimizer = optimizer, activation = activation,
                                  options = self.opt, plot = self.plot)

    def predict(self, X):
        return self.model.predict(X)
    
    

class QuantileNetworkModule(nn.Module):    
    def __init__(self, n_in, activation = 'ReLU', options = dict()):
        super(QuantileNetworkModule, self).__init__()
        
        self.n_in = n_in
        
        if activation == "ReLU":
            self.activation = nn.ReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid
        else:
            raise Exception(activation + "is currently not available as a activation function")
            
        # construct neural network model
        mdl_structure = []

        layers = [self.n_in] + [options['width']]*options['depth']
        if options['dropout_proportion'] == 0:
            for i in range(len(layers)-1):
                mdl_structure.append(nn.Linear(layers[i], layers[i+1]))
                mdl_structure.append(self.activation())
            mdl_structure.append(nn.Linear(layers[i+1], 1))
            
        else :
            for i in range(len(layers)-1):
                mdl_structure.append(nn.Linear(layers[i], layers[i+1]))
                mdl_structure.append(nn.Dropout(options['dropout_proportion']))
                mdl_structure.append(self.activation())
            mdl_structure.append(nn.Linear(layers[i+1], 1))

        self.fc_in = nn.Sequential(*mdl_structure)
        
    def forward(self, x):
        return self.fc_in(x)
    
    def predict(self, X):
        tX = torch.tensor(X, dtype = torch.float32)
        self.eval()
        self.zero_grad()
        yhat = self.forward(tX)[:,0]
        return yhat.data.numpy()




def fit_quantile(X, Y, tau = 0.5, nepochs=200, optimizer = 'SGD', activation = 'ReLU',
                 batch_size=64, options = dict(), plot = False, **kwargs):

    tX = torch.tensor(X, dtype = torch.float32)
    tY = torch.tensor(Y, dtype = torch.float32)
    train_ds = TensorDataset(tX, tY)
    train_dl = DataLoader(train_ds, batch_size, shuffle = True, drop_last=True)
    ttau = torch.tensor(tau, dtype = torch.float32)
    model = QuantileNetworkModule(tX.shape[1], activation = activation, options = options)

    # Save the model to file
    import pickle
    model_str = pickle.dumps(model)
    
    # Setup the SGD method
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=options['lr'], weight_decay=options['weight_decay'], 
                              nesterov=options['nesterov'], momentum=options['momentum'])
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['weight_decay'])
    else:
        raise Exception(options['optimizer'] + "is currently not available")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=options['lr_decay'])

    # Track progress
    train_losses = np.zeros(nepochs)

    # Univariate quantile loss
    def quantile_loss(yhat, y):
        z = y - yhat
        return torch.max(ttau[None]*z, (ttau[None] - 1)*z)

    epoch = 0
    loss_diff = 1

    while epoch < nepochs and loss_diff > options['tol'] :
        train_loss = torch.Tensor([0])
        for x_batch, y_batch in train_dl:
            model.train()
            model.zero_grad()
            yhat = model(x_batch)[:,0]
            loss = quantile_loss(yhat, y_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data

            if np.isnan(loss.data.numpy()):
                import warnings
                warnings.warn('NaNs encountered in training model.')
                break

        train_losses[epoch] = train_loss.data.numpy() / float(len(train_dl))
        scheduler.step()

        if epoch != 0:
            loss_diff = np.abs(train_losses[epoch] - train_losses[epoch-1])
        epoch += 1
    
    if plot == True:
        plt.plot(train_losses[2:epoch])
        plt.show()

    return model