#!/usr/bin/python

""" 
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import *
clf = AdaBoostClassifier()

# pred = clf.predict(features_test)
# accuracy  = accuracy_score(pred, labels_test)
# print accuracy
# features_train_new =  SelectPercentile(f_classif, percentile=1).fit_transform(features_train, labels_train)
# features_test_new =  SelectPercentile(f_classif, percentile=1).fit_transform(features_test, labels_test)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy  = accuracy_score(pred, labels_test)
print accuracy



#########################################################


