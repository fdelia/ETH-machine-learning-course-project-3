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

from sklearn.base import BaseEstimator, TransformerMixin
from biosppy.signals import ecg
import random
random.seed(99)

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
            self.hist_bins = [np.array([-1120.92739703,  -410.10302004,  -381.9367802 ,  -378.56527536,
        -174.60213944,   -44.34063315,   156.85575077,   267.69506558,
         339.42342893,   813.9679836 ,   973.56013152]), np.array([-1073.60184133,  -455.90183329,  -444.16046402,  -266.4309005 ,
        -136.41610024,    40.3241323 ,    82.42961269,   207.59071535,
         357.33894262,   797.85851412,  1104.19371508]), np.array([-1128.03563039,  -481.61721441,  -391.8380545 ,  -325.81087844,
        -100.55385922,    48.40905487,   145.29084922,   226.92310913,
         317.45573457,   679.60216005,  1042.88557759]), np.array([-975.7356248 , -552.58462036, -424.33531124, -244.59278393,
       -135.65844452,  -25.19848856,   65.99641195,  155.68679013,
        304.85466499,  524.97382413,  831.98856301]), np.array([ -964.60670382,  -477.64948448,  -385.7512008 ,  -315.21349457,
        -152.84926029,   -85.25289273,   156.42983286,   171.08830827,
         344.61128683,   632.58777948,  1012.74471941]), np.array([-1134.21396667,  -538.43695698,  -383.55196108,  -351.25518606,
        -234.1447024 ,   -76.85460874,    45.68341725,   179.81290443,
         283.45827579,   801.8892079 ,  1042.60967834]), np.array([ -992.1077973 ,  -543.54694461,  -375.95005102,  -303.90048148,
        -143.67198451,    56.16762597,    66.44392725,   207.85237126,
         380.28994569,   710.663395  ,  1041.77862555]), np.array([ -1.17059017e+03,  -4.53066930e+02,  -2.50693013e+02,
        -2.16122392e+02,  -1.31726746e+02,   1.13214948e+00,
         1.68241723e+02,   2.78649977e+02,   2.80757863e+02,
         7.83644496e+02,   9.68362007e+02]), np.array([-1087.66293359,  -549.50278667,  -408.46803696,  -277.20450193,
        -166.07720166,  -120.21777927,   127.31179008,   148.01407036,
         305.00966007,   518.83427404,   899.85406707]), np.array([-986.95299053, -503.2626519 , -433.34977809, -336.10288443,
       -183.63535388, -119.99037903,   96.05301919,  163.90039389,
        330.43404621,  765.67693595,  976.66567793]), np.array([-1099.08114631,  -528.03389226,  -335.17289441,  -212.82031621,
        -135.27839074,  -103.23247183,    58.56529893,   250.12959902,
         302.70520695,   752.02364255,  1007.3171732 ]), np.array([-1001.27139698,  -579.35034134,  -428.72446755,  -235.21167415,
        -126.52974159,  -118.11977088,    39.87136329,    74.37218865,
         244.81062741,   582.77849189,   872.47943575]), np.array([-1074.94606534,  -597.91369575,  -504.88658494,  -393.98020568,
        -130.40952056,   -74.93929184,    77.38499856,   255.65596205,
         288.87841233,   715.22901152,   975.36521957])]

        k=0
        for row in X:
            splits = np.array_split(row, int(self.splits))

            features = []
            for j, split in enumerate(splits):
                i = int(j / len(splits) * len(self.hist_bins))
                features.append(np.histogram(split, bins=self.hist_bins[i])[0])
                #features.append(np.histogram(split, bins=self.hist_bins)[0])

            try:
                e = ecg.ecg(row, show=False)
            except:
                e = {
                    "rpeaks": [], "heart_rate": [],
                    "templates_ts": [], "templates": []
                }
            rr = np.zeros(30)
            rr[:len(e['heart_rate'])] = e['heart_rate'][:30]
            features.append(rr)
            rr = np.zeros(50)
            rr[:len(e['templates_ts'])] = e['templates_ts'][:50]
            features.append(rr)

            features.append([len(e['rpeaks']), len(e['heart_rate']), len(e['templates'])])

            X_new.append(np.hstack(features))#.flatten())
            k+=1
            if k%1000 == 0:
                print("{} done".format(k))

        #print("features: "+str(len(X_new[0])))
        print(np.array(X_new).shape)
        return X_new


class Run(BaseEstimator, TransformerMixin):
    def __init__(self):
        pipe = Pipeline([
            ('BinsExtraction', RandomBinsExtraction(splits=80)),
            ('scaler', StandardScaler()),
            ('logreg',
                GradientBoostingClassifier(n_estimators=50, random_state=3)
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
