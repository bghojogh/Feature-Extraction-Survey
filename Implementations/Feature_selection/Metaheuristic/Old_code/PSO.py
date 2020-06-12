
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import pyswarms as ps
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
#import cantools
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def load(fname):
    lines = []
    with open(fname) as file:
        lines = file.readlines()
        lines = [x.strip() for x in lines] 

    flag = False
    X = []
    Y = []
    #q = []
    for line in lines:
        if ');' in line: flag = False
        if flag:
            case = line.split(',')
            y = case[0][1:].strip()
            t = case[len(case)-1]
            x = case[1:(len(case)-1)] + [t[:len(t)-1]]
            X.append(x)
            Y.append(y)
            #q.append(line)
        if 'cases = (' in line: flag = True
    X = np.array([[e.strip() for e in x] for x in X])
    Y = np.array(Y)
    
    cols_delete = []
    for col in range(X.shape[1]):
        if sum((X[:,col] == '?')*1) == X.shape[0]:
            cols_delete.append(col)

    X = np.delete(X, cols_delete, axis=1)
    X[X == '?'] = 0
    #ds = pd.DataFrame(X)
    #ds['target'] = pd.Series(Y)
    #ds['target-fac'] = pd.factorize(ds['target'])[0]
    #return ds
    return X,Y


# In[3]:


# ------------------------------------------------------------------------------------------------------
# Load and prep data
# ------------------------------------------------------------------------------------------------------


# In[4]:


train_X, train_y = load(fname='BreastCancer/breastCancer_Train.dbc')
test_X, test_y = load(fname='BreastCancer/breastCancer_Test.dbc')
print(train_X.shape)
print(test_X.shape)


# In[14]:


train_X


# In[13]:


print(sum((train_y == 'relapse')*1)*100/len(train_y))
print(sum((train_y == 'non-relapse')*1)*100/len(train_y))


# In[5]:


#model = linear_model.LogisticRegression()
model = svm.SVC(kernel='linear',C=1.0)

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
    total_features = train_X.shape[1]
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = train_X
    else:
        X_subset = train_X[:,m==1]
    # Perform classification and store performance in P
    model.fit(X_subset, train_y)
    #P = (model.predict(X_subset) == train_y).mean() # accuracy
    P = accuracy_score(train_y, model.predict(X_subset), normalize = True)
    # Compute for the objective function
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j


# In[6]:


def f(x, alpha=0.88):
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
    return np.array(j)


# In[7]:


# ------------------------------------------------------------------------------------------------------
# Run PSO
# ------------------------------------------------------------------------------------------------------


# In[8]:


# Create an instance of the classifier
#classifier = linear_model.LogisticRegression()

# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = train_X.shape[1] # dimensions should be the number of features
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, print_step=2, iters=10, verbose=2)


# In[13]:


print(sum((pos == 1)*1) / len(pos))
print(sum((pos == 1)*1))

