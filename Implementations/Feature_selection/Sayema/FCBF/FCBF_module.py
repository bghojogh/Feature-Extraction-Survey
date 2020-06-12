##https://github.com/SantiagoEG/FCBF_module
import numpy as np


def count_vals(x):
    vals = np.unique(x)
    occ = np.zeros(shape=vals.shape)
    for i in range(vals.size):
        occ[i] = np.sum(x == vals[i])
    return occ


def entropy(x):
    n = float(x.shape[0])
    ocurrence = count_vals(x)
    px = ocurrence / n
    return -1 * np.sum(px * np.log2(px))


def symmetricalUncertain(x, y):
    n = float(y.shape[0])
    vals = np.unique(y)
    # Computing Entropy for the feature x.
    Hx = entropy(x)
    # Computing Entropy for the feature y.
    Hy = entropy(y)
    # Computing Joint entropy between x and y.
    partial = np.zeros(shape=(vals.shape[0]))
    for i in range(vals.shape[0]):
        partial[i] = entropy(x[y == vals[i]])

    partial[np.isnan(partial) == 1] = 0
    py = count_vals(y).astype(dtype='float64') / n
    Hxy = np.sum(py[py > 0] * partial)
    IG = Hx - Hxy
    return 2 * IG / (Hx + Hy)


def suGroup(x, n):
    m = x.shape[0]
    x = np.reshape(x, (n, m / n)).T
    m = x.shape[1]
    SU_matrix = np.zeros(shape=(m, m))
    for j in range(m - 1):
        x2 = x[:, j + 1::]
        y = x[:, j]
        temp = np.apply_along_axis(symmetricalUncertain, 0, x2, y)
        for k in range(temp.shape[0]):
            SU_matrix[j, j + 1::] = temp
            SU_matrix[j + 1::, j] = temp

    return 1 / float(m - 1) * np.sum(SU_matrix, axis=1)


def isprime(a):
    return all(a % i for i in range(2, a))


"""
FCBF - Fast Correlation Based Filter
L. Yu and H. Liu. Feature Selection for High‐Dimensional Data: A Fast Correlation‐Based Filter Solution. 
In Proceedings of The Twentieth International Conference on Machine Leaning (ICML‐03), 856‐863.
Washington, D.C., August 21‐24, 2003.
"""


class FCBF:
    idx_sel = []

    def __init__(self, th=0):
        '''
        Parameters
        ---------------
            th = The initial threshold
        '''
        self.th = th

    def fit(self, x, y):
        '''
        This function executes FCBF algorithm and saves indexes
        of selected features in self.idx_sel

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''
        self.idx_sel = []
        """
        First Stage: Computing the SU for each feature with the response.
        """
        SU_vec = np.apply_along_axis(symmetricalUncertain, 0, x, y)
        SU_list = SU_vec[SU_vec > self.th]
        SU_list[::-1].sort()

        m = x[:, SU_vec > self.th].shape
        x_sorted = np.zeros(shape=m)

        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = 0
            x_sorted[:, i] = x[:, ind].copy()
            self.idx_sel.append(ind)

        """
        Second Stage: Identify relationships between feature to remove redundancy.
        """
        j = 0
        while True:
            """
            Stopping Criteria:The search finishes
            """
            if j >= x_sorted.shape[1]: break
            y = x_sorted[:, j].copy()
            x_list = x_sorted[:, j + 1:].copy()
            if x_list.shape[1] == 0: break

            SU_list_2 = SU_list[j + 1:]
            SU_x = np.apply_along_axis(symmetricalUncertain, 0,
                                       x_list, y)

            comp_SU = SU_x >= SU_list_2
            to_remove = np.where(comp_SU)[0] + j + 1
            if to_remove.size > 0:
                x_sorted = np.delete(x_sorted, to_remove, axis=1)
                SU_list = np.delete(SU_list, to_remove, axis=0)
                to_remove.sort()
                for r in reversed(to_remove):
                    self.idx_sel.remove(self.idx_sel[r])
            j = j + 1

    def fit_transform(self, x, y):
        '''
        This function fits the feature selection
        algorithm and returns the resulting subset.

        Parameters
        ---------------
            x = dataset  [NxM]
            y = label    [Nx1]
        '''
        self.fit(x, y)
        return x[:, self.idx_sel]

    def transform(self, x):
        '''
        This function applies the selection
        to the vector x.

        Parameters
        ---------------
            x = dataset  [NxM]
        '''
        return x[:, self.idx_sel]
