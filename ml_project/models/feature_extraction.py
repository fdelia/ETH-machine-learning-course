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

class BinsExtraction(BaseEstimator, TransformerMixin):
    """Build n bins with mean from values"""
    def __init__(self, bin_length=3, del_zero_std=False,
        images_x_from=False, images_x_to=False, more_features=False,
        images_y_from=False, images_y_to=False):
        self.bin_length = bin_length
        self.del_zero_std = del_zero_std
        self.more_features = more_features

        self.images_x_from = images_x_from
        self.images_x_to = images_x_to
        self.images_y_from = images_y_from
        self.images_y_to = images_y_to

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = []
        num_bins = int(len(X[0]) / self.bin_length) # int() same as floor()

        # Should raise error if file not found
        if self.del_zero_std:
            zero_std_ind = np.genfromtxt('/Users/fabiodelia/Dropbox/dev/ml-project/data/zero_std_ind_'+str(self.bin_length)+'.csv', delimiter=',')

        for row in X:
            # use only without "RemoveEmptyValues"
            # This is feature selection actually
            if self.images_x_from is not False and self.images_x_to is not False:
                #images = np.split(row, 176)[50:130] # pretty optimal already
                images = np.split(row, 176)[self.images_x_from : self.images_x_to]

                # x need to be set for this, but don't mind at the moment
                if self.images_y_from is not False and self.images_y_to is not False:
                    images_new = []
                    for image in images:
                        images_new.append(np.split(image, 208)[self.images_y_from : self.images_y_to])
                    images = np.array(images_new)

                row = np.array(images).flatten()
                num_bins = int(len(row) / self.bin_length)

            row2 = row[0 : num_bins*self.bin_length] # crop last elements, they are probably 0 anyway
            splits = np.split(row2, num_bins)
            # delete columns where std is zero
            if self.del_zero_std:
                splits = np.delete(splits, zero_std_ind, axis=0)

            features = np.mean(splits, axis=1)

            if self.more_features:
                splits = np.split(row2, self.bin_length)
                features2 = np.mean(splits, axis=0)
                X_new.append(np.concatenate((features, features2)))
            else:
                X_new.append(features)

        return X_new


class ImageExtraction(BaseEstimator, TransformerMixin):
    """Make use of spatial properties.
        First 14 and last 17 images are empty in all rows. Drop them.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = []
        SPLIT_BY = 2

        for row in X:
            images = np.split(row, 176)
            images_new = []

            # only get every nth element, not optimal, loosing information here
            for image in images[14:-17:4]:
                splits = np.split(image, 208 / SPLIT_BY * 176)
                image = np.mean(splits, axis=1)

                image = np.array(np.split(image, 208)).T
                splits = np.split(image.flatten(), 208 / SPLIT_BY * 176 / SPLIT_BY)
                image = np.mean(splits, axis=1)

                images_new.append(image)

            X_new.append(list(np.array(images_new).flatten()))

        return X_new
    #256 / 176
