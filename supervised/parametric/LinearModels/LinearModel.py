# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:41:49 2021

@author: jkcle
"""
# Import the f distribution from scipy.stats
from scipy.stats import f, t
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
    
    def _predict(self, X):
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
        super().__init__()
        self._TSS = None
        self._RSS = None
        self._MSS = None
        self.R_sq = None
        return
   
    def _calculate_model_stats(self, X, y):
        
        # Calculate the corrected total sum of squares (TSS).
        self._TSS = np.sum((y - np.mean(y))**2)       
        # Calculate the residual sum of squares (RSS).
        self._RSS = np.sum((y - self.predict(X))**2)
        # Calculate the model sum of squares (MSS).
        self._MSS = self._TSS - self._RSS
        # Calculate the R-squared of the fit model.
        self.R_sq = 1 - self._RSS / self._TSS
        return
    
    def predict(self, X):
        return self._predict(X)

class ClosedFormLinearModel(LinearRegression):
    """This is a parent class to those used for performing OLS and ridge 
    regression."""
    
    def __init__(self, add_intercept=True):
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
        super().__init__(add_intercept)
        return
    
    

    
    def _fit_svd(self, X, y, alpha=0.0):        
        """Estimates the coefficients of the OLS model using Singular Value
        Decomposition.
        
        Used the blog post at the following link:
            
        http://fa.bianp.net/blog/2011/ridge-regression-path/

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
        
        # Decompose X into U, s, and Vt
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        d = s / (s[:, np.newaxis].T ** 2 + alpha)
        # Calculate the coefficients minimizing the MSE with a penalty of
        # alpha on the l2 norm of the coefficients.
        self.beta_hat =  np.dot(d * U.T.dot(y), Vt).T
        self.beta_hat = np.dot(d*np.dot(np.transpose(U), y), Vt).reshape(1, 
                                                                         -1)[0]
        
        return


class OLS(ClosedFormLinearModel):
    """This class is used for performing OLS regression."""
    
    def __init__(self, add_intercept=True):
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
        super().__init__(add_intercept)
        self.df_model = None
        self.df_residuals = None
        self.F_stat = None
        self.F_prob = None
        self.beta_hat_se = None
        self.beta_hat_t_stats = None
        self.adj_R_sq = None
        self.sigma_hat = None
        return
    
    def fit(self, X, y):
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
            
        Returns
        -------
        None.

        """
        
        # Create a copy of X that has a column for the intercept if the user
        # wants one.
        X_copy = self._add_intercept(X)
        
        # Fit the model coefficients using SVD.
        self._fit_svd(X_copy, y, alpha=0.0)
        
        # Calculate model statistics.
        self._calculate_model_stats_ols(X, y)
        
        return
    
    def _calculate_model_stats_ols(self, X, y):
        
        # Calculate RSS, TSS, and R-sq
        self._calculate_model_stats(X, y)
        
        # Create a copy of X with an intercept column inserted at the 
        # beginning if the user desired it. 
        X_copy = self._add_intercept(X)
        
        # Estimate the sigma (standard deviation) of the response y.
        self.sigma_hat = np.sqrt(self._RSS 
                                 / (X_copy.shape[0] - X_copy.shape[1]))
        
        # Calculate the adjusted R-squares, which adjusts the R-square by 
        # penalizing the model for having variables which don't lower the
        # R-squared.
        self.adj_R_sq = (1 
                         - ((1 - self.R_sq)*(X_copy.shape[0] - 1))
                         /(X_copy.shape[0] - X_copy.shape[1]))
        
        # If the first column is all 1s for an intercept term, the degrees of
        # freedom of the model is the number of columns - 1.
        if sum(X_copy[:,0] == 1.) == X_copy.shape[0]:
            self.df_model = X_copy.shape[1] - 1
        # If there is an intercept column, the degrees of freedom of the model
        # is the number of columns.
        else:
            self.df_model = X_copy.shape[1]
        
        # Set the degrees of freedom of the error attribute for the model.
        self.df_residuals =  X_copy.shape[0] -  X_copy.shape[1]
        
        # Calculate the F-statistic for overall significance.
        self.F_stat = (self._MSS / self.df_model)/(self._RSS 
                                                   / self.df_residuals)
        # Calculate P(F-statistic) for overall significance.
        self.F_prob = 1. - f.cdf(self.F_stat, self.df_model, self.df_residuals)
        
        # Calculate the standard errors of the beta_hat coefficients.
        # First calculate the pseudo-inverse of XtX forcing all values to be
        # positive.
        XtX = np.matmul(np.transpose(X_copy), X_copy)
        XtX_pinv = np.absolute(np.linalg.pinv(XtX))
        # Calculate the standard errors of the coefficients.
        self.beta_hat_se = np.diagonal(np.sqrt(self.sigma_hat**2 * XtX_pinv))
        
        # Calculate the t-statitistics of the estimated coefficients.
        self.beta_hat_t_stats = self.beta_hat / self.beta_hat_se
        
        # Calculate the P(|t-stats| > 0) for the coefficients.
        self.beta_hat_prob = (1 
                              - t.cdf(np.absolute(self.beta_hat_t_stats), 
                                      df=self.df_residuals))*2
        
        return

class Ridge(ClosedFormLinearModel):
    """This class is used for performing Ridge regression."""
    
    def __init__(self, add_intercept=True, standardize=True):
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
        super().__init__(add_intercept)
        self.standardize = standardize
        self._y_bar = None
        self._X_bar = None
        self._X_std = None
        return
    
    def _standardize(self, X):
        if self.standardize:
            X_copy = (X - self._X_bar) / self._X_std
        else:
            X_copy = X
        return X_copy
        
        
    def fit(self, X, y, alpha=0.0):
        """
        This method estimates to coefficients of the OLS or ridge regression 
        model using singular value decomposition and calculates the attributes 
        that describe the fit of the model.

        Parameters
        ----------
        X : numpy ndarray
            A n x m matrix where the rows are observations and the columns are
            features used for predicting y.
        y : numpy ndarray
            A vector (numpy ndarray) of shape (n, ) of the response variable
            being predicted.
        alpha : float, optional
            The shrinkage or lambda to use for ridge regression. Will be zero
            for OLS. The default is 0.0.

        Returns
        -------
        None.

        """
        
        assert alpha >= 0.0, "alpha must be non-negative"
        
        self._y_bar = np.mean(y)
        self._X_bar = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        
        # Standardize X.
        X_copy = self._standardize(X)
        
        if self.add_intercept:
            demeaned_y = y - self._y_bar
        else: 
            demeaned_y = y - 0
        
        # Estimate the model coefficients using SVD.
        self._fit_svd(X_copy, demeaned_y, alpha)
        
        # Calculate model statistics.
        self._calculate_model_stats(X, y)
        
        return  
    
    def predict(self, X):
        X_copy = self._standardize(X)
        
        return self._y_bar + self._predict(X_copy)
    
    def _add_intercept(self, X):
        return X
        
        