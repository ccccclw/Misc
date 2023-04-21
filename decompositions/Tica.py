import numpy as np
import scipy.linalg
import warnings

class TICA():

    def __init__(self, n_ICs=None, lag_time=1, mapping = None):
        self.n_ICs = n_ICs
        self.lag_time = lag_time
        self.mapping = mapping
        
        self.n_features = None
        self.n_samples_ = None
        self._initialized = False

        #to calculate means
        # X[:-self.lag_time].sum(axis=0)
        self._sum_0_to_TminusTau = None
        # X[self.lag_time:].sum(axis=0)
        self._sum_tau_to_T = None
        # X[:].sum(axis=0)
        self._sum_0_to_T = None

        #to calculate covariances
        # X[:-self.lag_time].T dot X[self.lag_time:]
        self._outer_0_to_T_lagged = None
        # X[:-self.lag_time].T dot X[:-self.lag_time])
        self._outer_0_to_TminusTau = None
        # X[self.lag_time:].T dot X[self.lag_time:]
        self._outer_offset_to_T = None
        # store results of the eigendecompsition
        self._eigenvectors = None
        self._eigenvalues = None

    def _initialize(self, n_features):    #initialize all quantities
        if self.n_ICs is None:
            self.n_ICs = n_features
        self.n_features = n_features
        self.n_samples_ = 0
        self._sum_0_to_TminusTau = np.zeros(n_features)
        self._sum_tau_to_T = np.zeros(n_features)
        self._sum_0_to_T = np.zeros(n_features)
        self._outer_0_to_T_lagged = np.zeros((n_features, n_features))
        self._outer_0_to_TminusTau = np.zeros((n_features, n_features))
        self._outer_offset_to_T = np.zeros((n_features, n_features))
        self._initialized = True

    def _diagonalize(self):               #diagonalize the covariance matrix
        if self.n_samples_ == 0:
            raise RuntimeError('The model must be fit() before use.')

        lhs = self.lagged_covariance_
        rhs = self.covariance_

        if not np.allclose(lhs, lhs.T):
            raise RuntimeError('offset correlation matrix is not symmetric')
        if not np.allclose(rhs, rhs.T):
            raise RuntimeError('correlation matrix is not symmetric')

        vals, vecs = scipy.linalg.eigh(lhs, b=rhs,
            eigvals=(self.n_features-self.n_ICs, self.n_features-1))

        # sort eigenvalues in order of decreasing value
        ind = np.argsort(vals)[::-1]
        vals = vals[ind]
        vecs = vecs[:, ind]

        self._eigenvalues = vals
        self._eigenvectors = vecs


    def fit(self, X):                     #calculate matrix w/o-lagged used for covariance 
        self._initialize(X.shape[1])
        X = np.asarray(self._array2d(X), dtype=np.float64)

        if X.shape[1] > X.shape[0]:
            warnings.warn(f"The number of features ({X.shape[1]}) is greater than the number of samples ({X.shape[0]}).")

        if len(X) < self.lag_time:
            warnings.warn(f"length of data ({len(X)}) is too short for the lag time ({self.lag_time})")
            return

        self.n_samples_ += X.shape[0]

        self._outer_0_to_T_lagged += np.dot(X[:-self.lag_time].T, X[self.lag_time:])
        self._sum_0_to_TminusTau += X[:-self.lag_time].sum(axis=0)
        self._sum_tau_to_T += X[self.lag_time:].sum(axis=0)
        self._sum_0_to_T += X.sum(axis=0)
        self._outer_0_to_TminusTau += np.dot(X[:-self.lag_time].T, X[:-self.lag_time])
        self._outer_offset_to_T += np.dot(X[self.lag_time:].T, X[self.lag_time:])
        self._is_dirty = True

        return self

    def transform(self, X):               #project the input onto eigenvectors

        X = self._array2d(X)
        if self.means_ is not None:
            X = X - self.means_
        X_transformed = np.dot(X, self.components_.T)

        if self.mapping == 'kinetic':
            X_transformed *= self.eigenvalues_
        elif self.mapping == 'commute':
            regularized_timescales = 0.5 * self.timescales_ \
                                        * np.tanh(np.pi * ((self.timescales_ - self.lag_time) \
                                        / self.lag_time) + 1)
            X_transformed *= np.sqrt(regularized_timescales / 2)
            X_transformed = np.nan_to_num(X_transformed)
        elif self.mapping is not None:
            raise RuntimeError(f"{self.mapping} is not supported, use kinetic or commute.")
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def eigenvectors_(self):
        self._diagonalize()
        return self._eigenvectors[:, :self.n_ICs]

    @property
    def eigenvalues_(self):
        self._diagonalize()
        return self._eigenvalues_[:self.n_ICs]

    @property
    def timescales_(self):
        self._diagonalize()
        return -1. * self.lag_time / np.log(self._eigenvalues_[:self.n_ICs])

    @property
    def components_(self):
        return self.eigenvectors_[:, 0:self.n_ICs].T

    @property
    def means_(self):
        two_N = 2 * (self.n_samples_ - self.lag_time)
        means = (self._sum_0_to_TminusTau + self._sum_tau_to_T) / float(two_N)
        return means

    @property
    def lagged_covariance_(self):
        two_N = 2 * (self.n_samples_ - self.lag_time)
        term = (self._outer_0_to_T_lagged + self._outer_0_to_T_lagged.T) / two_N

        means = self.means_
        lagged_cov = term - np.outer(means, means)
        return lagged_cov

    @property
    def covariance_(self):
        two_N = 2 * (self.n_samples_ - self.lag_time)
        term = (self._outer_0_to_TminusTau + self._outer_offset_to_T) / two_N
        means = self.means_

        cov = term - np.outer(means, means)  # sample covariance matix

        return cov
    
    def _array2d(self, X, dtype=None, order=None, copy=False, force_all_finite=True):
        """Returns at least 2-d array with data from X"""
        X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
        if force_all_finite:
            self._assert_all_finite(X_2d)
        if X is X_2d and copy:
            X_2d = self._safe_copy(X_2d)
        return X_2d

    def _assert_all_finite(self, X):
        """Like assert_all_finite, but only for ndarray."""
        X = np.asanyarray(X)
        if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
                and not np.isfinite(X).all()):
            raise ValueError("Input contains NaN, infinity"
                            " or a value too large for %r." % X.dtype)

    def _safe_copy(self, X):
        # Copy, but keep the order
        return np.copy(X, order='K')
