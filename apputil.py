import pandas as pd


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

        # Make a copy of the dataframe
        X = X.copy()

        # Convert y to a pandas series 
        y = pd.Series(y)

        # y must not be missing any values
        #if y.isna().count() > 0:
        #    raise ValueError('Array y must not be missing any values.')

        # Get the column names in X before adding y
        X_columns = list(X.columns)
        print(X_columns)

        # Combine the data together
        X['y'] = y

        # Group by the columns in X and save the smaller data frame for use in predict
        self.X_grouped = X.groupby(X_columns, as_index=False).agg(self.estimate)

    def predict(self, X):
        return None


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