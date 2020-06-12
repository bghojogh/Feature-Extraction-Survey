import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv

class Kernel_FLDA:

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit_transform(self, X, y):
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        X = X.T
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n_samples = X.shape[1]
        labels_of_classes = list(set(y.ravel()))
        n_classes = len(labels_of_classes)
        # ------ M_*:
        print('M*....')
        M_star = np.zeros((n_samples, 1))
        for sample_index in range(n_samples):
            print('=========> sample: ', sample_index, ' out of ', n_samples, ' samples')
            x_total1 = X[:, sample_index]
            for sample_index_2 in range(n_samples):
                x_total2 = X[:, sample_index]
                M_star[sample_index] = M_star[sample_index] + (1 / n_samples) * self._radial_basis(xi=x_total1, xj=x_total2)
        # ------ M_c and M:
        print('M_c....')
        M = np.zeros((n_samples, n_samples))
        for class_index in range(n_classes):
            print('====> class: ', class_index)
            X_class = X_separated_classes[class_index]
            X_class = X_class.T
            n_samples_of_class = X_class.shape[1]
            M_c = np.zeros((n_samples, 1))
            for sample_index in range(n_samples):
                print('=========> sample: ', sample_index, ' out of ', n_samples, ' samples')
                x_total = X[:, sample_index]
                for sample_of_class_index in range(n_samples_of_class):
                    x_class = X_class[:, sample_of_class_index]
                    M_c[sample_index] = M_c[sample_index] + (1 / n_samples) * self._radial_basis(xi=x_total, xj=x_class)
            M = M + n_samples_of_class * (M_c - M_star).dot((M_c - M_star).T)
        # ------ N:
        print('N....')
        print(X_separated_classes[0].shape)
        print(X_separated_classes[9].shape)
        N = np.zeros((n_samples, n_samples))
        for class_index in range(n_classes):
            print('====> class: ', class_index)
            X_class = X_separated_classes[class_index]
            X_class = X_class.T
            n_samples_of_class = X_class.shape[1]
            K_c = np.zeros((n_samples, n_samples_of_class))
            for sample_index in range(n_samples):
                print('=========> sample: ', sample_index, ' out of ', n_samples, ' samples')
                x_total = X[:, sample_index]
                for sample_of_class_index in range(n_samples_of_class):
                    x_class = X_class[:, sample_of_class_index]
                    K_c[sample_index, sample_of_class_index] = self._radial_basis(xi=x_total, xj=x_class)
            N = N + K_c.dot(np.eye(n_samples_of_class) - (1/n_samples_of_class) * np.ones((n_samples_of_class, n_samples_of_class))).dot(K_c.T)
        # ------ embedding:
        eig_val, eig_vec = LA.eigh(inv(N).dot(M))
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            n_components = min(self.n_components, n_classes - 1)
            U = eig_vec[:, :n_components]
        else:
            U = eig_vec[:, :n_classes-1]
        n_samples_train = X.shape[1]
        n_samples_input = X.shape[1]
        K_t = np.zeros((n_samples_train, n_samples_input))
        for sample_train_index in range(n_samples_train):
            x_train = X[:, sample_train_index]
            for sample_input_index in range(n_samples_input):
                x_input = X[:, sample_input_index]
                K_t[sample_train_index, sample_input_index] = self._radial_basis(xi=x_train, xj=x_input)
        X_transformed = (U.T).dot(K_t)
        X_transformed = X_transformed.T
        return X_transformed

    def _build_kernel_matrix(self, X, kernel_func, option_kernel_func=None):  # --> K = self._build_kernel_matrix(X=X, kernel_func=self._radial_basis)
        # https://stats.stackexchange.com/questions/243104/how-to-build-and-use-the-kernel-trick-manually-in-python
        # X = X.T
        n_samples = X.shape[1]
        n_features = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            xi = X[:, i]
            for j in range(n_samples):
                xj = X[:, j]
                K[i, j] = kernel_func(xi, xj, option_kernel_func)
        return K

    def _radial_basis(self, xi, xj, gamma=None):
        if gamma is None:
            n_features = xi.shape[0]
            gamma = 1 / n_features
        r = (np.exp(-gamma * (LA.norm(xi - xj) ** 2)))
        return r

    def _separate_samples_of_classes(self, X, y):
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