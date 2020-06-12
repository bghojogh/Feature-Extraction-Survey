
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import pyswarms as ps
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score

import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from time import time
from PIL import Image
import glob
import re
from struct import *
from skimage.transform import resize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path_dataset_save = 'MNIST/10000Train_5000Test/'
file = open(path_dataset_save+'X_train_picked.pckl','rb')
X_train = pickle.load(file); file.close()
file = open(path_dataset_save+'y_train_picked.pckl','rb')
y_train = pickle.load(file); file.close()
file = open(path_dataset_save+'X_test_picked.pckl','rb')
X_test = pickle.load(file); file.close()
file = open(path_dataset_save+'y_test_picked.pckl','rb')
y_test = pickle.load(file); file.close()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[3]:


#model = linear_model.LogisticRegression()
#model = svm.SVC(kernel='linear',C=1.0)
#model = KNN(n_neighbors=1)
model = GaussianNB()

# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = X_train.shape[1]
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X_train
    else:
        X_subset = X_train[:,m==1]
    # Perform classification and store performance in P
    model.fit(X_subset, y_train)
    #P = (model.predict(X_subset) == train_y).mean() # accuracy
    P = accuracy_score(y_train, model.predict(X_subset), normalize = True)
    # Compute for the objective function
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    #print("j=" + str(j))
    return j


# In[4]:


def f(x, alpha=0.80):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    #print('n_particles=' + str(n_particles))
    #print('x=' + str(x.shape))
    #print(x[0])
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    #print(j)
    return np.array(j)


# In[5]:


# Create an instance of the classifier
#classifier = linear_model.LogisticRegression()
# c1 = cognitive parameter
# c2 = social parameter
# w = inertia paramter
# k = number of neighbors to be considered.
# p = 1 for L1 distance, 2 for L2 distance

# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}

# Call instance of PSO
dimensions = X_train.shape[1] # dimensions should be the number of features
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, print_step=1, iters=20, verbose=2)


# In[10]:


print('selected features = ' + str(sum((pos == 1)*1)) + '/' + str(len(pos)))
model.fit(X_train, y_train)
print('accuracy before FS = ' + str(accuracy_score(y_test, model.predict(X_test), normalize = True)*100))
X_subset = X_train[:,pos==1]
model.fit(X_subset, y_train)
print('accuracy after FS = ' + str(accuracy_score(y_test, model.predict(X_test[:,pos==1]), normalize = True)*100))

