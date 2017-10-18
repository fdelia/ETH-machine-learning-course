from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.svm import LinearSVR


class RandomEnsemble():
    """Build models with random parameters and average them"""
    def __init__(self):
        self.models = []

        hist_bins = [  1.00000000e+00,   2.43148939e+02 ,  5.52670768e+02,   6.63323565e+02, 7.82650357e+02 ,  1.02133853e+03 ,  1.08620935e+03,   1.20392913e+03, 1.39695300e+03 ,  1.75176484e+03]
        self.models.append(
            Pipeline([
                ('BinsExtraction', RandomBinsExtraction(splits=610,
                    images_x_from=50, images_x_to=132, hist_bins=hist_bins)),
                ('scaler', StandardScaler()),
                ('vct', VarianceThreshold(threshold=0.1)),
                ('linearSVR', LinearSVR(C=1.0, max_iter=1000))
            ])
        )

        hist_bins = [1, 282.10434686113894, 528.4350826042349, 635.7632261805744, 781.9301580581496, 962.0317275933281, 1079.2939329789033, 1246.3707282050862, 1393.0345691835053, 1721.8292294917992]
        self.models.append(
            Pipeline([
                ('BinsExtraction', RandomBinsExtraction(splits=610,
                    images_x_from=50, images_x_to=132, hist_bins=hist_bins)),
                ('scaler', StandardScaler()),
                ('vct', VarianceThreshold(threshold=0.1)),
                ('linearSVR', LinearSVR(C=1.0, max_iter=1000))
            ])
        )

    def fit(self, X, y=None):
        for i, model in enumerate(self.models):
            print("fitting model " + str(i))
            self.models[i].fit(X, y)
        return self

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        return np.mean(predictions, axis=0)




class RandomBinsExtraction(BaseEstimator, TransformerMixin):
    """Build n bins with mean from values"""
    def __init__(self, splits=610, hist_bins=None,
        images_x_from=50, images_x_to=130,
        images_y_from=0, images_y_to=204):
        self.splits = splits
        self.hist_bins = hist_bins

        self.images_x_from = images_x_from
        self.images_x_to = images_x_to
        self.images_y_from = images_y_from
        self.images_y_to = images_y_to

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = []
        split_x = 10
        split_y = 10
        if self.hist_bins is None:
            #self.hist_bins = [1, 282.10434686113894, 528.4350826042349, 635.7632261805744, 781.9301580581496, 962.0317275933281, 1079.2939329789033, 1246.3707282050862, 1393.0345691835053, 1721.8292294917992]
            self.hist_bins = [  1.00000000e+00,   2.43148939e+02 ,  5.52670768e+02,   6.63323565e+02,
   7.82650357e+02 ,  1.02133853e+03 ,  1.08620935e+03,   1.20392913e+03,
   1.39695300e+03 ,  1.75176484e+03]

        first = True
        for row in X:
            # use only without "RemoveEmptyValues"
            # This is feature selection actually
            if self.images_x_from is not False and self.images_x_to is not False:
                #images = np.split(row, 176)[50:130] # pretty optimal already
                images = np.split(row, 176)[self.images_x_from : self.images_x_to]

                # x needs to be set for this, but don't mind at the moment
                if self.images_y_from is not False and self.images_y_to is not False:
                    images_new = []
                    for image in images:
                        images_new.append(np.split(image, 208)[self.images_y_from : self.images_y_to])
                    images = np.array(images_new)

                row = np.array(images).flatten()
                #features = []
                #for image in images:
                #    for split in np.array_split(image, 104):
                #        features.append(np.histogram(split, bins=hist_bins, density=False)[0])

            splits = np.array_split(row, int(len(row) / self.splits))
            features = []
            for split in splits:
                features.append(np.histogram(split, bins=self.hist_bins)[0])

            X_new.append(np.array(features).flatten())
            if first:
                print("features: " + str(len(X_new[0])))
                first = False

        return X_new
