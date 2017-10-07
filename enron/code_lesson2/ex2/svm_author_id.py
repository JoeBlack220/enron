#!/usr/bin/python

""" 
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel="rbf",C=10000)
t0 = time()
clf.fit(features_train, labels_train)
print "\ntraining time:", round(time()-t0, 3),"s"

pred = clf.predict(features_test)

accuracy = accuracy_score(pred, labels_test)
print accuracy
sum = sum(pred)
print sum


#########################################################


