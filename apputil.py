import pandas as pd


def GroupEstimate(object):
    def __init__(self, estimate):
        """ Constructor

        Args:
            estimate (str): either 'mean' or 'median' 
        """
        if estimate != 'mean' and estimate != 'median':
            raise ValueError('Estimate must be either mean or median.')

        self.estimate = estimate
    
    def fit(self, X, y):
        return None

    def predict(self, X):
        return None