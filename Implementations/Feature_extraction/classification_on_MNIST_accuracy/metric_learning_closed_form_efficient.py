import numpy as np
from numpy import linalg as LA

class Metric_learning_closed_form:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None

    def fit_transform(self, X, y):
        self.fit(X, y)
        X_transformed = self.transform(X, y)
        return X_transformed

    def fit(self, X, y):
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # ------ M_S:
        # print('M_S....')
        n_classes = len(X_separated_classes)
        scatter = np.zeros((n_features, n_features))
        S_cardinality = 0
        for class_index in range(n_classes):
            # print('====> class: ', class_index)
            X_class = X_separated_classes[class_index]
            scatter = scatter + self.efficient_scatter(X=X_class)
            S_cardinality = S_cardinality + X_class.shape[0]
        M_S = (1 / S_cardinality) * scatter

        # ------ M_D:
        # print('M_D....')
        scatter = np.zeros((n_features, n_features))
        D_cardinality = 0
        for class_index in range(n_classes-1):
            # print('====> class: ', class_index)
            # ----- don't consider previous classes:
            X_PreviousClassesRemoved = X
            y_PreviousClassesRemoved = y
            if class_index != 0:
                for previous_class_index in range(class_index-1):
                    mask = np.asarray(y_PreviousClassesRemoved) == previous_class_index
                    X_PreviousClassesRemoved = X_PreviousClassesRemoved[~mask, :]
                    y_PreviousClassesRemoved = y_PreviousClassesRemoved[~mask]
            mask = np.asarray(y_PreviousClassesRemoved) == class_index
            X_this_class = X_PreviousClassesRemoved[mask, :]
            X_other_classes = X_PreviousClassesRemoved[~mask, :]
            subset_of_D = np.zeros((X_other_classes.shape[0] + 1, n_features))
            subset_of_D[0:-1, :] = X_other_classes
            n_samples_of_class = X_this_class.shape[0]
            for sample_index in range(n_samples_of_class):
                subset_of_D[-1, :] = X_this_class[sample_index, :]
                scatter = scatter + self.efficient_scatter(X=subset_of_D)
                D_cardinality = D_cardinality + subset_of_D.shape[0]
        M_D = (1 / D_cardinality) * scatter

        # ------ embedding:
        eig_val, eig_vec = LA.eigh(M_S - M_D)
        idx = (-eig_val).argsort()[::-1]   # sort eigenvalues in ascending order (smallest eigenvalue first)
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

    def _separate_samples_of_classes(self, X, y):
        # X --> rows: samples, columns: features
        y = np.asarray(y)
        y = y.reshape((-1, 1))
        yX = np.column_stack((y, X))
        yX = yX[yX[:, 0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
        y = yX[:, 0]
        X = yX[:, 1:]
        labels_of_classes = list(set(y))
        number_of_classes = len(labels_of_classes)
        dimension_of_data = X.shape[1]
        X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        class_index = 0
        index_start_new_class = 0
        n_samples = X.shape[0]
        for sample_index in range(1, n_samples):
            if y[sample_index] != y[sample_index - 1] or sample_index == n_samples-1:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
                index_start_new_class = sample_index
                class_index = class_index + 1
        return X_separated_classes

    def efficient_scatter(self, X, compute_repatative_pairs=False, weights=None):
        # X --> rows: samples, columns: features
        # https://stackoverflow.com/questions/31145918/fast-weighted-scatter-matrix-calculation
        # https://stackoverflow.com/questions/27627896/fast-differences-of-all-row-pairs-with-numpy
        if weights is None:
            n_samples = X.shape[0]
            weights = np.ones((n_samples, n_samples))
        scatter = np.tensordot(weights.sum(1)[:, None] * X, X, axes=[(0,), (0,)])
        scatter += np.tensordot(weights.sum(0)[:, None] * X, X, axes=[(0,), (0,)])
        scatter -= np.tensordot(np.dot(weights, X), X, axes=[(0,), (0,)])
        scatter -= np.tensordot(X, np.dot(weights, X), axes=[(0,), (0,)])
        if compute_repatative_pairs is False:
            scatter = 0.5 * scatter
        return scatter