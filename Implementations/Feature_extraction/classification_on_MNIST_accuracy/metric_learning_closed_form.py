import numpy as np
from numpy import linalg as LA

class Metric_learning_closed_form:

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit_transform(self, X, y):
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        X = np.transpose(X)
        # ------ M_S:
        number_of_classes = len(X_separated_classes)
        dimension_of_data = X.shape[0]
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        S_cardinality = 0
        for class_index in range(number_of_classes):
            X_class = X_separated_classes[class_index]
            X_class = np.transpose(X_class)
            cardinality_of_class = X_class.shape[1]
            for sample_index in range(cardinality_of_class-1):
                x_i = X_class[:, sample_index]
                for sample_index_2 in range(sample_index+1, cardinality_of_class):
                    x_j = X_class[:, sample_index_2]
                    scatter = scatter + (x_i - x_j).dot(np.transpose((x_i - x_j)))
                    S_cardinality = S_cardinality + 1
        M_S = (1 / S_cardinality) * scatter
        # ------ M_D:
        scatter = np.zeros((dimension_of_data, dimension_of_data))
        D_cardinality = 0
        for class_index in range(number_of_classes-1):
            X_class = X_separated_classes[class_index]
            X_class = np.transpose(X_class)
            cardinality_of_class = X_class.shape[1]
            for sample_index in range(cardinality_of_class):
                x_i = X_class[:, sample_index]
                for class_index_other in range(class_index+1, number_of_classes):
                    X_class_other = X_separated_classes[class_index_other]
                    X_class_other = np.transpose(X_class_other)
                    cardinality_of_class_other = X_class_other.shape[1]
                    for sample_index_other in range(cardinality_of_class_other):
                        x_j = X_class_other[:, sample_index_other]
                        scatter = scatter + (x_i - x_j).dot(np.transpose((x_i - x_j)))
                        D_cardinality = D_cardinality + 1
        M_D = (1 / D_cardinality) * scatter
        # ------ embedding:
        eig_val, eig_vec = LA.eigh(M_S - M_D)
        idx = (-eig_val).argsort()[::-1]   # sort eigenvalues in ascending order (smallest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            U = eig_vec[:, :self.n_components]
        else:
            U = eig_vec
        X_transformed = np.transpose(U).dot(X)
        X_transformed = np.transpose(X_transformed)
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