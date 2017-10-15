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


from sklearn.cluster import KMeans, MiniBatchKMeans
import random
random.seed(42)
class ClusteredHistExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, n_samples=2, images_x_from=False, images_x_to=False):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.images_x_from = images_x_from
        self.images_x_to = images_x_to


    def cutImage(self, x):
        if self.images_x_from is not False and self.images_x_to is not False:
            #images = np.split(row, 176)[50:130] # pretty optimal already
            side_images = np.split(x, 176)[self.images_x_from : self.images_x_to]
            x = np.array(side_images).flatten()
        return x


    def fit(self, X, y=None):
        samples = random.sample(list(X), self.n_samples)
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state=42)
        # self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=100, random_state=42)

        centers = []
        for i, sample in enumerate(samples):
            # samples[i] = sample[1672390 : -786303]
            sample = self.cutImage(sample)

            samples[i] = sample[(sample > 0) & (sample < 1800)]
            # self.kmeans.fit(np.array([samples[i]]).T)
            # centers.append(np.sort(np.array(self.kmeans.cluster_centers_).flatten()))
            # print(str(i) + ' done')

        # if True: # use all centers
        #     values = np.array(centers).flatten()
        #     values = np.sort(values)
        # else: # take means of centers
        #     values = np.mean(centers, axis=0)


        # compute cluster centers
        self.kmeans.fit(np.array(samples).T)
        values = self.kmeans.cluster_centers_.T
        print('fitted')

        # mean of the clusters over the rows
        for i, v in enumerate(values.T):
            values.T[i] = np.sort(v)

        values = np.mean(values.T, axis=0)



        self.edges = [1] # leave out 0
        for center_1, center_2 in zip(values[:-1], values[1:]):
            self.edges.append(.5 * (center_1 + center_2))

        print('n edges: ' + str(len(self.edges)))
        return self

    def transform(self, X, y=None):
        # np.histogram to make bins from edges, counts the number of pixels
        X_new = []
        for x in X:
            x = self.cutImage(x)
            hist = np.histogram(x, bins=self.edges)
            X_new.append(hist[0])

        return X_new


class GivenEdgesExtraction(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.edges = [1,
 175,
 190.0,
 200,
 210,
 220,
 230.0,
 240,
 250.29576676838963,
 260.59153353677925,
 270.7998907955566,
 281.00824805433393,
 296.51672876785585,
 312.4192910985714,
 328.7096455492857,
 345,
 365.11915911105507,
 385.23831822211014,
 421.68407741265736,
 458.1298366032046,
 510.43647791541645,
 576.5739902509565,
 609.5574644576008,
 626.0492015609229,
 642.540938664245,
 655.0834926946815,
 666.5424184427528,
 680,
 688.7915018861073,
 697.5830037722146,
 705.1660075444291,
 712.7490113166436,
 720.3320150888583,
 730.1660075444291,
 740,
 747.7243484748634,
 755.4486969497268,
 763.1730454245902,
 770.8973938994536,
 778.3608893423097,
 785.8243847851659,
 792.7855130483395,
 799.7466413115131,
 807.5810902304434,
 816.5985296073651,
 825.6159689842868,
 836.8877682054391,
 848.1595674265914,
 863.9400863362044,
 879.7206052458175,
 905.8611747488183,
 932.0017442518191,
 950.2522817644435,
 968.5028192770679,
 995.0990029527295,
 1021.6951866283912,
 1045.622987987501,
 1084.7830302704533,
 1123.9430725534057,
 1151.1392141568033,
 1178.3353557602009,
 1203.4127942147347,
 1228.4902326692686,
 1253.1926742146657,
 1265.5438949873642,
 1277.8951157600625,
 1288.9475578800311,
 1300,
 1309.8438400564696,
 1319.6876801129395,
 1329.5315201694093,
 1339.375360225879,
 1347.7457855765365,
 1356.1162109271943,
 1365.7868972366239,
 1375.4575835460535,
 1386.4285308142548,
 1400,
 1408.3298109092373,
 1416.6596218184745,
 1424.9894327277118,
 1433.319243636949,
 1442,
 1448.5,
 1454.875,
 1461.25,
 1470.625,
 1480,
 1490.0,
 1500,
 1509.375,
 1518.75,
 1530,
 1546.25,
 1562.5,
 1581.25,
 1610,
 1700.0]
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # double number of bins
        # new = []
        # for i in range(len(self.edges)):
        #     new.append(self.edges[i])
        #     if i < len(self.edges)-1:
        #         new.append(0.5 * (self.edges[i] + self.edges[i+1]))
        #
        # new2 = []
        # for i in range(len(new)):
        #     new2.append(new[i])
        #     if i < len(new)-1:
        #         new2.append(0.5 * (new[i] + new[i+1]))


        X_new = []
        for x in X:
            # X_new.append(np.histogram(x, bins=new2)[0])
            X_new.append(np.histogram(x, bins=self.edges)[0])
        return X_new

# Build n bins from hist/distribution of values
# Number of values between x, x+1
class HistExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, bins=10):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # make stats here
        bins = self.bins
        X_new = []
        # wichtige teile: 400-600, 600-900, 900-1200, 1300-1500
        steps = [100, 200, 400, 600, 900, 1200, 1300, 1500, 1700]

        for x in X:
            row_new = []
            # for split in np.split(x, 1):

            step = 5
            low = 200
            for high in range(low+step, 400, step):
                row_new.append(((x >= low) & (x <= high)).sum())
                low = high

            low = 400
            for high in range(low+step, 600, step):
                row_new.append(((x >= low) & (x <= high)).sum())
                low = high

            low = 900
            for high in range(low+step, 1200, step):
                row_new.append(((x >= low) & (x <= high)).sum())
                low = high

            low = 1300
            for high in range(low+step, 1500, step):
                row_new.append(((x >= low) & (x <= high)).sum())
                low = high

            # mean: 0.861744557643  std: 0.00949954149053
            # for i, low in enumerate(steps[:-1]):
            #     high = steps[i+1]
            #     row_new.append(((x >= low) & (x <= high)).sum())

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

                # x needs to be set for this, but don't mind at the moment
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
