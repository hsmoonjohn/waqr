import numpy as np
# import os
# import sys
# import torch
# import matplotlib.pyplot as plt
# import time
from DeepQuantile import DQR
from sklearn.linear_model import LinearRegression
'git on google drive'

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

    def fit_ls(self, tau1,  weighting = 'es'):
        qnet1 = DQR(self.X, self.Y, options=self.opt)
        if weighting == 'es':
            qnet1.fit(tau=tau1)
        elif weighting == 'sq':
            qnet1.fit(tau=1-tau1)
        qy1 = qnet1.predict(self.X)
        r1 = self.pseudo_outcome(qy1=qy1, tau1=tau1, weighting=weighting)
        return LinearRegression(fit_intercept=False).fit(self.X, r1)












