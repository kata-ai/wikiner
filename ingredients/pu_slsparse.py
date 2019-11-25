import numpy as np
import scipy as sp
# from scipy.linalg import solve_banded

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics import pairwise_distances

import pywsl.utils.comcalc as com
from pywsl.pul.pu_mr import calc_risk
from pywsl.utils import check


class SparsePU_SL(BaseEstimator, ClassifierMixin):
    def __init__(self, prior=.5, sigma=.1, lam=1, basis='gauss', n_basis=200, sparse=True):
        check.in_range(prior, 0, 1, name="prior")
        self.prior = prior
        self.basis = basis
        self.sigma = sigma
        self.lam = lam
        self.n_basis = n_basis
        self.coef_ = None
        self._x_c = None
        self._sparse = sparse
#            if self.sigma is None:
#                d_u = com.squared_dist(x_u, self._x_c)
#                self.sigma = np.sqrt(np.median(d_u))

    def fit(self, x, y):
        check_classification_targets(y)
        # x, y = check_X_y(x, y, y_numeric=True)
        x, y = check_X_y(x, y, accept_sparse=True)
        x_p, x_u = x[y == +1, :], x[y == 0, :]
        n_p, n_u = x_p.shape[0], x_u.shape[0]

        if self.basis == 'gauss':
            b = np.minimum(n_u, self.n_basis)
            center_index = np.random.permutation(n_u)[:b]
            self._x_c = x_u[center_index, :]
        elif self.basis == 'lm':
            b = x_p.shape[1] + 1
        else:
            raise ValueError('Invalid basis type: {}.'.format(self.basis))

        k_p, k_u = self._ker(x_p), self._ker(x_u)

        H = k_u.T.dot(k_u)/n_u
        h = 2*self.prior*np.mean(k_p, axis=0) - np.mean(k_u, axis=0)
        R = self.lam*np.eye(b)

        if self._sparse:
            # print(sp.sparse.issparse(H))
            # print(sp.sparse.issparse(R))
            # print(sp.sparse.issparse(H + R))
            # print(h.shape)
            sA = sp.sparse.csr_matrix(H+R)
            h = np.expand_dims(h, axis=-1)
            # print(h.shape)
            sb = sp.sparse.csr_matrix(h)
            # print(sb.shape)
            self.coef_ = sp.sparse.linalg.spsolve(sA, sb)
        else:
            A = H + R
            b = h
            self.coef_ = sp.linalg.solve(A, b)

        return self


    def predict(self, x):
        check_is_fitted(self, 'coef_')
        x = check_array(x, accept_sparse=self._sparse)
        return np.sign(.1 + np.sign(self._ker(x).dot(self.coef_)))


    def score(self, x, y):
        x_p, x_u = x[y == +1, :], x[y == 0, :]
        k_p, k_u = self._ker(x[y == +1, :]), self._ker(x[y == 0, :])
        g_p, g_u = k_p.dot(self.coef_), k_u.dot(self.coef_)
        pu_risk = calc_risk(g_p, g_u, self.prior)
        return 1 - pu_risk


    def _ker(self, x):
        if self.basis == 'gauss':
            if self._sparse:
                K = com.gauss_basis(pairwise_distances(x, self._x_c), self.sigma)
            else:
                K = com.gauss_basis(com.squared_dist(x, self._x_c), self.sigma)
        elif self.basis == 'lm':
            if sp.sparse.issparse(x) and self._sparse:
                # convert to dense
                x = x.toarray()
            K = com.homo_coord(x)
        return K
