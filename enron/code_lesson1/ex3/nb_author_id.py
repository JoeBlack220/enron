#!/usr/bin/python

""" 
 	Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

clf = GaussianNB()
time0 = time()
clf.fit(features_train,labels_train)
time1 = time() - time0
print "The training consumes %f" %(time1) + "s"
time3 = time()
clf.predict(features_test)
time4 = time() - time3
print "The predicting consumes %f" %(time4) + "s"


#########################################################
