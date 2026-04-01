import pandas as pd
import numpy as np


class GroupEstimate(object):
    def __init__(self, estimate='mean'):
        """ Constructor

        Args:
            estimate (str): either 'mean' or 'median' 
        """
        if estimate != 'mean' and estimate != 'median':
            raise ValueError('Estimate must be either mean or median.')

        self.estimate = estimate
    
    def fit(self, X, y):
        """ Adds the data in y as a column in X

        Args:
            X (pandas.DataFrame): a dataframe with the data to analyze
            y (array-like): the data to fit to X
        """

        # Make a copy of the dataframe
        X = X.copy()

        # Convert y to a pandas series 
        y = pd.Series(y)

        # Get the column names in X before adding y
        X_columns = list(X.columns)

        # Combine the data together
        X['y'] = y

        # Group by the columns in X and save the smaller data frame for use in predict
        self.X_grouped = X.groupby(X_columns, as_index=False).agg(self.estimate)

    def predict(self, X):
        """ Returns the predicted value for various categories given in X

        Args:
            X (list(list)): a 2D list of categories matching the columns in the original X 

        Returns:
            list: the predicted values for each entry in X 
        """
        # The list to add the estimates to that we will return
        estimates = []

        # Since we don't know what the columns are in X, we can filter by one
        # column at a time
        X_columns = self.X_grouped.columns
        for category in X:
            filter = self.X_grouped
            for i in range(len(category)):
                filter = filter.loc[self.X_grouped[X_columns[i]] == category[i]]

            # If the filtered dataframe is empty, add NaN to the list
            if filter.empty:
                estimates.append(np.nan)

            # Otherwise, add the y value
            else:
                # Get the individual y value out of the series
                # (Converting to float to match example in exercises)
                estimates.append(float(filter['y'].iloc[0]))

        return np.array(estimates)


# For testing
if __name__ == '__main__':
    df_raw = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/coffee_analysis.csv')

    X = df_raw[["loc_country", "roast"]]
    y = df_raw["rating"]

    gm = GroupEstimate(estimate='mean')
    gm.fit(X, y)

    X_ = [["Guatemala", "Light"],
        ["Mexico", "Medium"],
        ["Canada", "Dark"]]

    print(gm.predict(X_))