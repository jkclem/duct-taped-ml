# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:41:49 2021

@author: jkcle
"""
import numpy as np

class LinearModel():
    """"""
    def __init__(self, add_coefficient):
        self.add_coefficient = add_coefficient
        self.beta_hat = None
    
    def predict(self, X):
        
        # If this object needs to add an intercept to new data, add one.
        if self.add_coefficient == True:
            # Create an array of 1s equal in length to the observations in X.
            intercept_column = np.repeat(1, repeats=X.shape[0])
            # Insert it at the 0-th column index.
            X_copy = np.insert(X, 0, intercept_column, axis=1)
        # Otherwise, just copy X.
        else:
            X_copy = X
        
        # Forget X to free memory.
        del X
        
        return np.matmul(X_copy, self.beta_hat)