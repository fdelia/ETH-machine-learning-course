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

# Build n bins from hist/distribution of values
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

# Build n bins with mean from values
class BinsExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, bin_length=3, del_zero_std=False):
        self.bin_length = bin_length
        self.del_zero_std = del_zero_std

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = []
        num_bins = int(len(X[0]) / self.bin_length) # int() same as floor()

        # Should raise error if file not found
        if self.del_zero_std:
            zero_std_ind = np.genfromtxt('../../data/zero_std_ind_'+str(self.bin_length)+'.csv', delimiter=',')
        # bins = 50
        # l = int(1600/bins)

        for row in X:
            row = row[0 : num_bins*self.bin_length] # crop last elements, they are probably 0 anyway
            splits = np.split(row, num_bins)
            # X_new.append(np.concatenate([np.mean(splits, axis=1), np.std(splits, axis=1)]))
            # s = []
            # for i in range(0, bins):
            #     s.append( (((i*l) < row) & (row <= (i+1)*l)).sum() )
            # X_new.append(np.concatenate([np.mean(splits, axis=1), s]))

            # delete columns where std is zero
            if self.del_zero_std:
                splits = np.delete(splits, zero_std_ind, axis=0)

            X_new.append(np.mean(splits, axis=1))

        return X_new
