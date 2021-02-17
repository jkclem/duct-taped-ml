import numpy as np

class KNN:
    """This class is the parent class of the KNNClassifier and
    KNNRegresser classes."""
    
    def __init__(self, k, weights="uniform"):
        """Initialize the KNN class with a k and weighting option ("uniform", or
        "distance").

        Parameters
        ----------
        k : int
            k is a positive integer and the number of nearest neighbors to use
            for predicting the label of a new observation.
        weights : str, optional
            This parameter sets how to weight the labels of the k-nearest 
            neighbors. Options are "uniform" for equal weighting, or 
            "distance" for inverse-distance weighting. The default is 
            "uniform".

        Returns
        -------
        None.

        """
        
        # Check that k is a positive integer.
        assert (type(k) == int) & (k > 0), "k should be a positive integer."
        # Check that weights is either 'uniform' or 'distance'.
        assert ((weights == "uniform") 
                | (weights 
                   == "distance")), "weights should be 'uniform' or 'distance'."
        
        # Initialize the k, which is the number of nearest neighbors.
        self.k = k
        # Initialize the type of weighting of the data points ("uniform", 
        # or "distance").
        self.weights = weights
        # Initialize the X and y attributes with None.
        self._X = None
        self._y = None
        
    def fit(self, X, y):
        """This method saves the training data so it can be used to make 
        predictions for new data.

        Parameters
        ----------
        X : numpy ndarray
            A n x m matrix containing the features used to predict y.
        y : numpy ndarray
            A numpy array of shape (n, ) of the labels of the observations in
            X.

        Returns
        -------
        None.

        """
        
        # Check that X and y are numpy arrays.
        assert ((type(X) == np.ndarray) 
                & (type(y) == np.ndarray)), "X and y must by numpy arrays."
        # Check that X is a 2-D array with at least 1 column.
        assert len(X.shape) == 2, "X must by an n by m array, where m >= 1."
        # Check that y is a 1-D array.
        assert len(y.shape) == 1, "y must be a 1-D array"
        # Check that the number of rows in X matches elements in y.
        assert (y.shape[0] 
                == X.shape[0]), "X must have as many rows as y has elements."
        
        self._X = X
        self._y = y.astype('object')
        
        return
    
    def _euclidean_distances(self, array, k_nearest_indices=None):
        """"""
        
        # If an array of indices is not passed, calculate the Euclidean
        # distance between the input array and the entire training data
        # set.
        if type(k_nearest_indices) == type(None):
            euclidean_distances = np.sqrt(np.sum((self._X-array)**2, 
                                                 axis=-1))
        # If an array of the indicies of the k nearest data points is passed
        # to the method from the _predict_one method, calculate the Euclidean 
        # distances of the k nearest points.
        else:
            euclidean_distances = np.sqrt(np.sum((self._X[k_nearest_indices]
                                                  -array)**2, 
                                                 axis=-1))
        # Return the Euclidean distances of the array from the data points.
        return euclidean_distances
    
    def _k_nearest_indices(self, array):
        """This method takes in an array and finds the indices of the k 
        nearest data points in the training data."""
        
        # Calculate the Euclidean distances between the new data point and 
        # the training data.
        euclidean_distances = self._euclidean_distances(array)
        # Sort the indices from smallest Euclidean distance from the new
        # data point to the largest.
        sorted_indices = np.argsort(euclidean_distances, 
                                    axis=None, 
                                    kind="quicksort")
        # Save the indices of the k nearest points.
        k_nearest_indices = sorted_indices[0:self.k]
        # Return the indicies of the k nearest points.
        return k_nearest_indices
    
    def _predict_one(self, array):
        """This method will be overwritten in the child classes. It predicts
        the label for a single observation. In the predict method, it will 
        be applied to each row of the array to predict for multiple
        observations."""
        
        return
    
    def predict(self, array, smoothing=0, probability=False):
        """This method predicts the label for each row in an array of new
        observations by applying the _predict_one method to each row. The
        smoothing factor is set to 0, but can be increased if divide by 0
        warnings are encountered. The probability argument is only used
        if passed as true to the KNNClassifier class."""
        
        # Check that the smoothing factor is non-negative.
        assert ((smoothing >= 0) 
                & (smoothing < 1)), "0 <= smoothing < 1"
        
        # If there is only one observation in the new array, use the
        # _predict_one method to calculate a prediction.
        if len(self._X[0].shape) == len(array.shape):
            predictions = self._predict_one(array, 
                                            smoothing=smoothing,
                                            probability=probability)
        
        else:
            # Check that the input array has the same number of columns as X.
            assert (array.shape[1] 
                    == self._X.shape[1]), ("Input array must have the same ",
                                           "number of columns as X.")
            
            # Apply the _predict_one method to each row in the input array.
            predictions = np.apply_along_axis(self._predict_one,
                                              axis=1, 
                                              arr=array, 
                                              smoothing=smoothing,
                                              probability=probability)
        # Return the predictions.
        return predictions


class KNNClassifier(KNN):
    """This class is the child class of the KNN class that is performs
    KNN classification."""
    
    def _predict_one(self, array, smoothing, probability):
        """This method predicts the label for a single observation and
        is applied to the rows of an input array in the predict method
        to predict multiple observations."""
        
        # Retrieve the indices of the k nearest observations to the new
        # data point.
        k_nearest_indices = self._k_nearest_indices(array)
        # Retrieve the labels of the k nearest observations to the new
        # data point.
        k_nearest_labels = self._y[k_nearest_indices]
        
        # If using uniform weighting, create a weight array of k ones. 
        if self.weights == "uniform":
            my_weights = np.repeat(1, repeats=self.k)
            
        # If using inverse distance weighting...    
        else:
            # Retrieve the sorted Euclidean distances of the k nearest points.
            k_nearest_distances = self._euclidean_distances(array,
                                                            k_nearest_indices=k_nearest_indices)
            # Set the weights for the training labels equal to the inverse 
            # of their features' Euclidean distance from the new data point.
            my_weights = 1 / (k_nearest_distances + smoothing)
            
        # Get the unique labels in the k nearest points.
        label_set = np.unique(k_nearest_labels)
        # Create an empty list to hold the weighted vote tallys
        weighted_vote_tally = []
        # Create a dictionary to hold the probabilities of belonging to each 
        # class.
        prob_dict = dict()
        
        # For each label in the k nearest points...
        for label in label_set:
            # Create a binary mask to select the weights associated with the 
            # label.
            temp_indexes = k_nearest_labels==label
            # Sum the weights associated with the label and append it to the 
            # list.
            weighted_vote = np.sum(my_weights[temp_indexes])
            weighted_vote_tally.append(weighted_vote)
            # Create a key from the label and make the value the weighted vote 
            # share.
            prob_dict[label] = weighted_vote / np.sum(my_weights)
        
        # Find the index of highest number of votes in weighted_vote_tally.
        most_likely_class_index = np.argmax(weighted_vote_tally)
        # Use that index to grab the label with the highest vote share.
        label = label_set[most_likely_class_index]
        
        # If the user wants probabilities, return the dictionary.
        if probability == True:
            return prob_dict
        # If the user doesn't want predicted classes, return the predicted 
        # label.
        else:
            # Return the label.
            return label

    
class KNNRegresser(KNN):
    """This class is the child class of the KNN class that is performs
    KNN regression."""
    
    def _predict_one(self, array, smoothing, probability):
        """This method predicts the label for a single observation and
        is applied to the rows of an input array in the predict method
        to predict multiple observations. The probability argument has 
        nothing to do with the KNNRegresser. It is only included to 
        allow the KNN parent class to have a predict method that can
        be used with the two child classes, making maintenance easier."""
        
        # Retrieve the indices of the k nearest observations to the new
        # data point.
        k_nearest_indices = self._k_nearest_indices(array)
        # Retrieve the labels of the k nearest observations to the new
        # data point.
        k_nearest_labels = self._y[k_nearest_indices]
        
        # If using uniform weighting, create a weight array of k ones. 
        if self.weights == "uniform":
            my_weights = np.repeat(1, repeats=self.k)
        # If using inverse distance weighting...
        else:
            # Retrieve the sorted Euclidean distances of the k nearest points.
            k_nearest_distances = self._euclidean_distances(array,
                                                            k_nearest_indices=k_nearest_indices)
            # Set the weights for the training labels equal to the inverse 
            # of their features' Euclidean distance from the new data point.
            my_weights = 1 / (k_nearest_distances + smoothing)
        
        # Multiply the weights by their associated labels, sum the products
        # and divide by the sum of the weights.
        label = np.matmul(my_weights, k_nearest_labels) / np.sum(my_weights)
        
        # Return the label.
        return label