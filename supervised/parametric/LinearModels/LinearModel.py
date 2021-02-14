# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:41:49 2021

@author: jkcle
"""
import numpy as np

class LinearModel():
    """The Linear Model Class is the parent class to all linear models."""
    
    def __init__(self, add_coefficient=True):
        """
        Initializes the class with a boolean indicating whether or not the
        class needs to add a column of 1s to all feature matrices to fit an
        intercept and an empty beta_hat vector that will hold the regression
        model's coefficients.
        
        Parameters
        ----------
        add_coefficient : bool, optional
            Tells the class if it needs to add a column of 1s in the first
            column of any data set passed to it, for fitting or prediction. If
            the user does not want to include an intercept in the model, or 
            has already included a column of 1s in the data set for the 
            intercept, this should be set to False. The default is True.

        Returns
        -------
        None.

        """
        self.add_coefficient = add_coefficient
        self.beta_hat = None
        return
    
    def fit():
        """This method will be overwritten by each of its child classes 
        because the method of fitting the linear model will vary from
        algorithm to algorithm.
        """
        pass
    
    def predict(self, X):
        """This method predicts the response values of the input array, X, in 
        the scale the model is estimated in; e.g. a logistic model will return
        predictions in log-odds. The columns of X must match the number of 
        columns on the array on which the model was fit. The ordering must be
        identical as well for the predictions to mean anything.

        Parameters
        ----------
        X : numpy ndarray
            A n x m matrix, where the n rows represent observations and the m
            columns represent features of the observations.

        Returns
        -------
        numpy ndarray
            Returns a numpy ndarray with n elements that are the predicted 
            values of the response for each observation in X.

        """
        
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
        
        # Return the predictions.
        return np.matmul(X_copy, self.beta_hat)
    
class LinearRegression(LinearModel):
    """This class serves as the parent class to the OLS, LAD, LASSO, Ridge, 
    and Elastic Net regression classes."""
    
    def __init__(self, add_coefficient):
        """
        Initializes the class with a boolean indicating whether or not the
        class needs to add a column of 1s to all feature matrices to fit an
        intercept and an empty beta_hat vector that will hold the regression
        model's coefficients. Initialized attributes for the corrected total
        sum of squares and residual sum of squares that will be used to 
        calculate the R-squared and adjusted R-squared attributes.
        
        Parameters
        ----------
        add_coefficient : bool, optional
            Tells the class if it needs to add a column of 1s in the first
            column of any data set passed to it, for fitting or prediction. If
            the user does not want to include an intercept in the model, or 
            has already included a column of 1s in the data set for the 
            intercept, this should be set to False. The default is True.

        Returns
        -------
        None.

        """
        self.add_coefficient = add_coefficient
        self.beta_hat = None
        self._TSS = None
        self._RSS = None
        self.R_sq = None
        return
   
    def _calculate_R_sq(self):
        # Define a method to calculate the R-squared of the model.
        self.R_sq = 1 - self._RSS / self._TSS
        return
    
class OLS(LinearRegression):
    def __init__(self, *args, **kwargs):
        super(OLS, self).__init__(*args, **kwargs)
        return
    
    def fit(self, X, y, method="qr"):
        if method == "qr":
            self._fit_qr(X, y)
        elif method == "moore-penrose inverse":
            self._fit_pinv(X, y)
        else:
            self._fit_svd(X, y)
        
        self._TSS = np.sum((y - np.mean(y))**2)
        y_hat = self.predict(X)
        self._RSS = np.sum((y - y_hat)**2)
        self.R_sq = 1 - self._TSS / self._RSS
        
        return
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        