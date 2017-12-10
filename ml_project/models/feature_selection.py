from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement


class NonZeroSelection(BaseEstimator, TransformerMixin):
    """Select non-zero voxels"""
    def fit(self, X, y=None):
        X = check_array(X)
        self.nonzero = X.sum(axis=0) > 0

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["nonzero"])
        X = check_array(X)
        return X[:, self.nonzero]


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

class RandomBinsExtraction(BaseEstimator, TransformerMixin):
    """Build n bins with mean from values"""
    def __init__(self, splits=100, hist_bins=None):
        self.splits = splits
        self.hist_bins = hist_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = []
        if self.hist_bins is None:
            self.hist_bins = [-1063.23640194, -502.03339032,  -422.5276436,   -286.42136028,   -97.20735832,   -55.16501241,  123.49800188 ,  206.09874563 ,  337.38048621,   637.69219956,   939.23230402]

        for row in X:
            splits = np.array_split(row, int(self.splits))

            features = []
            for j, split in enumerate(splits):
                i = int(j / len(splits) * len(self.hist_bins))
                #features.append(np.histogram(split, bins=self.hist_bins[i])[0])
                features.append(np.histogram(split, bins=self.hist_bins)[0])

            X_new.append(np.array(features).flatten())

        #print("features: "+str(len(X_new[0])))
        return X_new


class Run(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipe = None

    def sampleIt(self, X, y):
        X_new = []
        y_new = []
        sample_w = []
        for i, row_y in enumerate(y):
            for cl, p_i in enumerate(row_y):
                X_new.append(X[i])
                y_new.append(cl)
                sample_w.append(p_i)
        return X_new, y_new, sample_w

    def fit(self, X, y=None):
        if y is not None:
            X, y, weights = self.sampleIt(X, y)

        pipe = Pipeline([
            ('BinsExtraction', RandomBinsExtraction(splits=10000)),
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(
                C=1.0, solver='newton-cg', n_jobs=-1
            ))
        ])
        pipe.fit(X, y, **{'logreg__sample_weight': weights})
        self.pipe = pipe
        return self

    def transform(self, X, y=None):
        return X

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def predict(self, X):
        return self.pipe.predict(X)
