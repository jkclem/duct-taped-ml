# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:25:41 2021

@author: jkcle
"""

import numpy as np

class PCA:
    """This class performs principal components analysis for finding 
    orthogonal linear combinations of the columns of a matrix for 
    dimensionality reduction. It has methods to fit (and standardize) input 
    data, set the number of components to keep, and transform new data 
    (standardizing with the mean and standard deviation of the 'training' data.
    It has attributes like variance_explained to help the user to choose the 
    number of components to use for transforming data."""
    
    def __init__(self):
        # will hold all of the principal components, with the columns as the 
        # loading for the ordinal variables and the rows for the principal 
        # components
        self.all_components = np.empty(0)
        # will hold the n_components number of components
        self.components = np.empty(0)
        # will hold a vector of the variance explained by each component
        self.variance_explained = np.empty(0)
        # will hold a vector of the ratio of variance explained to total 
        # variance by each component
        self.ratio_var_explained = np.empty(0)
        # will hold a vector of the cumulative ratio of variance explained to 
        # total variance by each component
        self.cumulative_ratio_var_explained = np.empty(0)
        # sets the number of principal components in self.components
        self.n_components = 0
        # will remember the training data's means of each column
        self.x_bar = np.empty(0)
        # will remember the training data's standard deviation of each column
        self.std_dev = np.empty(0)
        # remember if input data needs to be standardized
        self.standardize = False
        
    def fit(self, X, standardize=True):
        """Takes in an NumPy array or Pandas DataFrame, standardizes it to 
        have mean 0 and standard deviation 1 if the user desires, and finds 
        the rotation matrix, as well as the variance explained by each 
        principal component, as well as the ratio of total variance explained 
        by a principal component, and the cumulative ratio of variance 
        explained by each principal component. It saves each of those with 
        their corresponding attribute of the class.

        Parameters
        ----------
        X : numpy ndarray
            A n x m matrix where the rows are columns and the columns are
            features.
        standardize : bool, optional
            If desired, standardize the data prior to performing PCA. 
            The default is True.

        Returns
        -------
        None.

        """
        
        assert type(standardize) == bool, "standardize must be True or False"
        
        # copy the input data
        X_copy = X.copy()
        
        # remember the mean and standard deviation for transforming data if 
        # standardize is True
        self.x_bar = X_copy.mean()
        self.std_dev = X_copy.std()
        
        # if the data needs standardization, standardize it
        if standardize:
            
            # remember if input data needs to be standardized
            self.standardize = True
            
            # de-mean the data
            X_copy -= self.x_bar
            # divide each column by its std dev
            X_copy /= self.std_dev
            
        # calculate the sample covariance matrix
        S = np.cov(X_copy.T)
        
        # get the eigenvalues and eigenvectors from the sample covariance 
        # matrix
        eig_vals, eig_vecs = np.linalg.eig(S)
        
        # get the sorted indexes of the eigenvalues (largest to smallest)
        sorted_indexes = np.argsort(eig_vals)[::-1]
        # sort the eigenvalues from largest to smallest
        eig_vals = eig_vals[sorted_indexes]
        # sort the eigenvectors based on their eigen values
        eig_vecs = eig_vecs[:, sorted_indexes]
        
        # if the data was standardized, total variance is the number of 
        # variables
        if standardize:
            total_var = S.shape[0]
            
        # otherwise use the trace of the sample covariance matrix, aka sum of 
        # all variances
        else:
            total_var = np.trace(S)
            
        # calculate the percent of variance explained by each principal 
        # component to 6 decimal places
        ratio_var_explained = (eig_vals / total_var).round(6)
        
        # calculate the cumlative  percent of variance explained by each 
        # principal component to 6 
        # decimal places
        cumu_ratio_var_explained = ratio_var_explained.cumsum()
        # set the last element to 100 because it should be 100, but there is
        # probably floating point error
        cumu_ratio_var_explained[-1] = 1
        
        # now set the attributes with the appropriate values
        self.all_components = eig_vecs.T
        self.components = eig_vecs.T
        self.variance_explained = eig_vals
        self.ratio_var_explained = ratio_var_explained
        self.cumulative_ratio_var_explained = cumu_ratio_var_explained
        self.n_components = S.shape[0]
        
        # end the fit function
        return
    
    def keep_n_components(self, n_components):
        """Sets the number of components to keep if the user does not want to 
        keep them all and modifies the number components making up the 
        rotation matrix.

        Parameters
        ----------
        n_components : int
            Sets the number of principal components to keep for transforming
            the data.

        Returns
        -------
        None.

        """
        
        # make sure n_components is a positive integer less than or equal to 
        # the total number of components
        assert ((type(n_components) == int) & 
                (n_components > 0) & 
                (n_components 
                 <= self.all_components.shape[0])), 'n_components must be a positive integer less than or equal to the number of variables in X'
        
        # reset the number of components and select n_components
        self.n_components = n_components
        self.components = self.all_components[0: self.n_components]
            
        # end the function
        return
    
    def transform(self, X):
        """Takes in an NumPy array or Pandas DataFrame, standardizes it to 
        have mean 0 and standard deviation 1 if the user desires, and returns 
        the transformed data set based on the rotation matrix with 
        n_components principal components."""
              
        # copy the input data
        X_copy = X.copy()
        
        # if the data needs standardization, standardize it
        if self.standardize:
            # de-mean the data
            X_copy -= self.x_bar
            # divide each column by its std dev
            X_copy /= self.std_dev
        
        # returned the transformed data
        return X_copy @ self.components.T