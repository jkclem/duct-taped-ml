# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:24:37 2021

@author: jkcle
"""
import numpy as np

def make_datasets(size, cols, num_zero=0, beta_scale=1., intercept=1., eta=1., seed=1):
    np.random.seed(seed)
    
    X = np.random.normal(loc=0, scale=1, size=(size,cols))
    beta = np.random.normal(loc=0, scale=beta_scale, size=cols)
    
    if num_zero != 0.:
        beta_indices = np.arange(0, beta.shape[0], 1)
        zero_indices = np.random.choice(beta_indices, num_zero, replace=False)
        beta[zero_indices] = 0
    
    y = intercept + np.matmul(X, beta) + np.random.normal(loc=0, 
                                                          scale=eta, 
                                                          size=size) 
    X_train = X[:round(size/2),:]
    y_train = y[:round(size/2)]  
    X_test = X[round(size/2):,:]
    y_test = y[round(size/2):] 
    
    return X_train, y_train, X_test, y_test