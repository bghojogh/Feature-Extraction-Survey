import numpy as np
from numpy import linalg as LA

class Supervised_PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None

    def fit_transform(self, X, y):
        self.fit(X, y)
        X_transformed = self.transform(X, y)
        return X_transformed

    def fit(self, X, y):
        X = X.T
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n = X.shape[1]
        H = np.eye(n) - ((1/n) * np.ones((n,n)))
        B = (y.T).dot(y)
        eig_val, eig_vec = LA.eigh( X.dot(H).dot(B).dot(H).dot(X.T) )
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            self.U = eig_vec[:, :self.n_components]
        else:
            self.U = eig_vec

    def transform(self, X, y=None):
        X_transformed = ((self.U).T).dot(X.T)
        X_transformed = X_transformed.T
        return X_transformed