# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:41:49 2021

@author: jkcle
"""
import numpy as np

class LinearModel():
    """The Linear Model Class is the parent class to all linear models."""
    
    def __init__(self, add_intercept=True):
        """
        Initializes the class with a boolean indicating whether or not the
        class needs to add a column of 1s to all feature matrices to fit an
        intercept and an empty beta_hat vector that will hold the regression
        model's coefficients.
        
        Parameters
        ----------
        add_intercept : bool, optional
            Tells the class if it needs to add a column of 1s in the first
            column of any data set passed to it, for fitting or prediction. If
            the user does not want to include an intercept in the model, or 
            has already included a column of 1s in the data set for the 
            intercept, this should be set to False. The default is True.

        Returns
        -------
        None.

        """
        self.add_intercept = add_intercept
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
        
        X_copy = self._add_intercept(X)
        
        # Return the predictions.
        return np.matmul(X_copy, self.beta_hat)
    
    def _add_intercept(self, X):
        # If this object needs to add an intercept to new data, add one.
        if self.add_intercept == True:
            # Create an array of 1s equal in length to the observations in X.
            intercept_column = np.repeat(1, repeats=X.shape[0])
            # Insert it at the 0-th column index.
            X_copy = np.insert(X, 0, intercept_column, axis=1)
        # Otherwise, just copy X.
        else:
            X_copy = X
        
        return X_copy
    
class LinearRegression(LinearModel):
    """This class serves as the parent class to the OLS, LAD, LASSO, Ridge, 
    and Elastic Net regression classes."""
    
    def __init__(self, add_intercept=True):
        """
        Initializes the class with a boolean indicating whether or not the
        class needs to add a column of 1s to all feature matrices to fit an
        intercept and an empty beta_hat vector that will hold the regression
        model's coefficients. Initialized attributes for the corrected total
        sum of squares and residual sum of squares that will be used to 
        calculate the R-squared attribute.
        
        Parameters
        ----------
        add_intercept : bool, optional
            Tells the class if it needs to add a column of 1s in the first
            column of any data set passed to it, for fitting or prediction. If
            the user does not want to include an intercept in the model, or 
            has already included a column of 1s in the data set for the 
            intercept, this should be set to False. The default is True.

        Returns
        -------
        None.

        """
        self.add_intercept = add_intercept
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
    """This class is used for performing OLS regression."""
    
    def __init__(self, *args, **kwargs):
        """
        Initializes the class with a boolean indicating whether or not the
        class needs to add a column of 1s to all feature matrices to fit an
        intercept and an empty beta_hat vector that will hold the regression
        model's coefficients. Initialized attributes for the corrected total
        sum of squares and residual sum of squares that will be used to 
        calculate the R-squared and adjusted R-squared attributes.
        
        Parameters
        ----------
        add_intercept : bool, optional
            Tells the class if it needs to add a column of 1s in the first
            column of any data set passed to it, for fitting or prediction. If
            the user does not want to include an intercept in the model, or 
            has already included a column of 1s in the data set for the 
            intercept, this should be set to False. The default is True.

        Returns
        -------
        None.

        """
        self.adj_R_sq = None
        self.sigma_hat = None
        super(OLS, self).__init__(*args, **kwargs)
        return
    
    def fit(self, X, y, method="qr"):
        """
        This method estimates to coefficients of the OLS model and calculates
        the attributes that describe the fit of the model.

        Parameters
        ----------
        X : numpy ndarray
            A n x m matrix where the rows are observations and the columns are
            features used for predicting y.
        y : numpy ndarray
            A vector (numpy ndarray) of shape (n, ) of the response variable
            being predicted.
        method : str, optional
            Decides how the OLS fit is estimated. The OLS coefficients can be
            estimated by either using QR factorization ("qr"), the 
            Moore-Penrose  pseudo-inverse of XtX^-1 ("moore-penrose"), or 
            Singular Value Decomposition "svd". The default is "qr".

        Returns
        -------
        None.

        """
        
        # Create a copy of X that has a column for the intercept if the user
        # wants one.
        X_copy = self._add_intercept(X)
        
        # Fit the model coefficients using QR factorization if the user wants.
        if method == "qr":
            self._fit_qr(X_copy, y)
        # Fit the model coefficients using the Moore-Penrose psuedo-inverse of
        # XtX^-1 if the user wants.
        elif method == "moore-penrose":
            self._fit_pinv(X_copy, y)
        # Fit the model coefficients using SVD if the user wants.
        else:
            self._fit_svd(X_copy, y)
        
        # Calculate model statistics.
        self._calculate_model_stats(X, y)
        
        return
    
    def _fit_qr(self, X, y):
        """Estimates the coefficients of the OLS model using QR factorization.

        Parameters
        ----------
        X : numpy ndarray
            A n x m matrix where the rows are observations and the columns are
            features used for predicting y.
        y : numpy ndarray
            A vector (numpy ndarray) of shape (n, ) of the response variable
            being predicted.

        Returns
        -------
        None.

        """
        # Factorize X into Q, an orthonormal matrix, and R, an upper 
        # triangular matrix, such that X = QR.
        Q, R = np.linalg.qr(X)
        # Multiply the transpose of Q and y.
        z = np.matmul(np.transpose(Q), y)
        # Set the beta_hat attribute as the estimate of beta vector.
        self.beta_hat = np.linalg.solve(R, z)
        
        return
    
    def _fit_pinv(self, X, y):
        """Estimates the coefficients of the OLS model using the normal 
        equation, but substituting the Moore-Penrose pseudo-inverse of XtX^-1 
        instead of directly calculating XtX^-1.

        Parameters
        ----------
        X : numpy ndarray
            A n x m matrix where the rows are observations and the columns are
            features used for predicting y.
        y : numpy ndarray
            A vector (numpy ndarray) of shape (n, ) of the response variable
            being predicted.

        Returns
        -------
        None.

        """
        XtX = np.matmul(np.transpose(X), X)
        XtX_pinv = np.linalg.pinv(XtX)
        XtX_pinv_Xt = np.matmul(XtX_pinv, np.transpose(X)) 
        self.beta_hat = np.matmul(XtX_pinv_Xt, y)
        
        return
    
    def _calculate_model_stats(self, X, y):
        
        # Create a copy of X with an intercept column inserted at the 
        # beginning if the user desired it. 
        X_copy = self._add_intercept(X)
        
        # Calculate the corrected total sum of squares (TSS).
        self._TSS = np.sum((y - np.mean(y))**2)
        
        # Calculate the residual sum of squares (RSS).
        self._RSS = np.sum((y - self.predict(X))**2)
        
        # Calculate the R-squared of the fit model.
        self.R_sq = 1 - self._RSS / self._TSS
        
        # Calculate the adjusted R-squares, which adjusts the R-square by 
        # penalizing the model for having variables which don't lower the
        # R-squared.
        self.adj_R_sq = (1 
                         - ((1 - self.R_sq)*(X_copy.shape[0] - 1))
                         /(X_copy.shape[0] - X_copy.shape[1]))
        
        # Estimate the sigma (standard deviation) of the response y.
        self.sigma_hat = np.sqrt(self._RSS 
                                 / (X_copy.shape[0] - X_copy.shape[1]))
        
        return

size = 100
np.random.seed(1)
X1 = np.random.normal(loc=0, scale=1, size=size)       
X2 = np.random.normal(loc=0, scale=1, size=size)   
X3 = np.random.normal(loc=0, scale=1, size=size) 
X = np.concatenate([X1, X2, X3]).reshape(size, 3)
y = 1 + 2*X[:,0] + 3*X[:,1] + 4*X[:,2] + np.random.normal(loc=0, scale=1, 
                                                          size=size)
my_ols = OLS()
my_ols.fit(X, y, method="moore-penrose")        
        
        
        
        
        
        
        
        
        
        
        