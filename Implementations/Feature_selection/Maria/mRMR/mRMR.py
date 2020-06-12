# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 01:20:37 2018

@author: Samad
"""

import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from skfeature.function.information_theoretical_based import MRMR


def main():
    # load training data
    mat0 = scipy.io.loadmat('fea.mat')    
    X = mat0['fea_tr']    # data
    X = X.astype(float)
    
    mat1 = scipy.io.loadmat('gnd.mat')
    y = mat1['gnd_tr']    # label
    
    n_samples, n_features = X.shape    # number of samples and number of features
    
    #load test data
    mat2 = scipy.io.loadmat('fea_t.mat')    
    X_t = mat2['fea_tst']    # data
    X_t = X_t.astype(float)
    
    mat3 = scipy.io.loadmat('gnd_t.mat')
    y_t = mat3['gnd_tst']    # label
    
    n_samples_t, n_features_t = X_t.shape    # number of samples and number of features

    # perform evaluation on classification task
    num_fea = 400    # number of selected features
   
    gnb = GaussianNB()

    idx,_,_ = MRMR.mrmr(X, y, n_selected_features=num_fea)

    # obtain the index of each feature on the training set
    idx,_,_ = MRMR.mrmr(X, y, n_selected_features=num_fea)

    # obtain the dataset on the selected features
    features = X[:, idx[0:num_fea]]

    # train a classification model with the selected features on the training dataset
    gnb.fit(features, y)
    
    # obtain the dataset on the selected features of the test set for prediction purposes
    features_t = X_t[:, idx[0:num_fea]]
    
    # predict the class labels of test data
    y_predict = gnb.predict(features_t)

    # obtain the classification accuracy on the test data
    acc = accuracy_score(y_t, y_predict)
    
    # output the average classification accuracy over all 10 folds
    print 'Accuracy:', float(acc)/10

if __name__ == '__main__':
    main()