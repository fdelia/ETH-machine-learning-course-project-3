from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


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
            self.hist_bins = [np.array([-1083.59921692,  -476.33763822,  -391.50489743,  -355.60320737,
        -158.43665552,   -14.84714956,    94.77981205,   218.64467678,
         346.5553962 ,   734.50802575,  1002.02972199]), np.array([-1019.66220701,  -523.43519633,  -379.07005136,  -331.66828082,
        -156.15875706,    18.12736061,    39.05190812,   258.69304456,
         372.82923485,   730.21415049,  1041.77824411]), np.array([-1092.05133913,  -467.36450973,  -323.77895088,  -276.7100964 ,
        -177.65586773,   -29.1918839 ,    95.14627871,   267.45422342,
         319.30367525,   752.0415106 ,  1031.95221106]), np.array([-1026.0900565 ,  -550.04804846,  -487.69032655,  -306.6158703 ,
        -166.89823442,   -61.82025277,   108.66079029,   124.07144747,
         274.9071339 ,   519.71265528,   911.63967546]), np.array([-1012.82800316,  -526.8370896 ,  -425.31935619,  -369.87368713,
        -134.25612587,  -125.16553721,   137.63442111,   221.84134438,
         308.374747  ,   698.18557519,  1044.20382967])]

        for row in X:
            splits = np.array_split(row, int(self.splits))

            features = []
            for j, split in enumerate(splits):
                i = int(j / len(splits) * len(self.hist_bins))
                # features.append(np.histogram(split, bins=self.hist_bins)[0])
                features.append(np.histogram(split, bins=self.hist_bins[i])[0])

            X_new.append(np.array(features).flatten())
        return X_new


class Run(BaseEstimator, TransformerMixin):
    def __init__(self):
        pipe = Pipeline([
            ('BinsExtraction', RandomBinsExtraction(splits=80)),
            ('scaler', StandardScaler()),
            ('logreg',
                GradientBoostingClassifier(n_estimators=30, learning_rate=0.1)
             )
        ])
        self.pipe = pipe

    def fit(self, X, y=None):
        self.pipe.fit(X, y)
        return self

    def transform(self, X, y=None):
        return X

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def predict(self, X):
        return np.array(self.pipe.predict(X)).astype(int)
