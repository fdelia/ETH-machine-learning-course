from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np

class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new

# Build n bins for hist of values
# Number of values between x, x+1
class StatsExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, bins=10):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # make stats here
        bins = self.bins
        X_new = []
        l = int(1600/bins)

        for x in X:
            row_new = [
                # np.mean(x),
                # np.std(x),
                # np.var(x),
                # np.max(x),
                # np.count_nonzero(x),
            ]

            for i in range(0, bins):
                row_new.append( (((i*l) < x) & (x <= (i+1)*l)).sum() )

            X_new.append(row_new)

        return np.array(X_new)

# Build n bins with mean for values
class BinsExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, bin_length=3):
        self.bin_length = bin_length

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = []
        l = self.bin_length

        for row in X:
            row_new = []
            for i in range(0, int(len(row) / l) - 1):
                row_new.append(np.mean(row[i*l : (i+1)*l]))
            X_new.append(row_new)

        return np.array(X_new)
