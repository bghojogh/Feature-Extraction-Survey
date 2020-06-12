
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

import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms
import sys

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


def avg(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))


def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)

        # apply classification algorithm
        #clf = LogisticRegression()
        clf = GaussianNB()
        #clf.fit(X_subset, y)
        #return accuracy_score(y, clf.predict(X_subset), normalize = True)
        return (avg(cross_val_score(clf, X_subset, y, cv=2)),)
    else:
        return(0,)


def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    for individual in hof:
        #print(individual.fitness.values)
        #print(maxAccurcy)
        if(individual.fitness.values[0] > maxAccurcy):
            maxAccurcy = individual.fitness.values
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


def getArguments():
    """
    Get argumments from command-line
    If pass only dataframe path, pop and gen will be default
    """
    dfPath = sys.argv[1]
    if(len(sys.argv) == 4):
        pop = int(sys.argv[2])
        gen = int(sys.argv[3])
    else:
        pop = 10
        gen = 2
    return dfPath, pop, gen


# In[4]:


hof = geneticAlgorithm(pd.DataFrame(X_train), y_train, 30, 20)
accuracy, individual, header = bestIndividual(hof, pd.DataFrame(X_train), y_train)
print('Best Accuracy: \t' + str(accuracy))
print('Number of Features in Subset: \t' + str(individual.count(1)))
print('Individual: \t\t' + str(individual))
print('Feature Subset\t: ' + str(header))


# In[12]:


model = GaussianNB()
print('selected features = ' + str(len(header)) + '/' + str(X_train.shape[1]))
model.fit(X_train, y_train)
print('accuracy before FS = ' + str(accuracy_score(y_test, model.predict(X_test), normalize = True)*100))
X_subset = X_train[:,header]
model.fit(X_subset, y_train)
print('accuracy after FS = ' + str(accuracy_score(y_test, model.predict(X_test[:,header]), normalize = True)*100))

