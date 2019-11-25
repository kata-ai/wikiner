
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer

from ingredients.pu_slsparse import SparsePU_SL
# from pywsl.pul.pu_mr import PU_SL

class EntityDetectionPU(ClassifierMixin):
    def __init__(self, prior=.5, sigma=.1, lam=1, basis='gauss', n_basis=200):
        self.clf = SparsePU_SL(prior=prior, sigma=sigma, lam=lam, basis=basis, n_basis=n_basis)
        # self.clf = PU_SL(prior=prior, sigma=sigma, lam=lam, basis=basis, n_basis=n_basis)
        # self.clf = SVC()
        self.featureizer = DictVectorizer(sparse=True)

    def fit(self, X, y):
        x_feat = self.featureizer.fit_transform(X)
        self.clf.fit(x_feat, y)
        return self

    def predict(self, X):
        x_feat = self.featureizer.transform(X)
        return self.clf.predict(x_feat)

    def predict_sent(self, X_sent):
        for xi in X_sent:
            yield(self.predict([xi]))
