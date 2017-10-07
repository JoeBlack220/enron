#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
import numpy
from outlier_cleaner import outlierCleaner
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
from sklearn.cross_validation import train_test_split
target, features = targetFeatureSplit( data )
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

#print data
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, target_train)
print reg.score(feature_train, target_train), reg.coef_

try:
    plt.plot(features, reg.predict(features), color="blue")
except NameError:
    pass
plt.scatter(features, target)
plt.show()


### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(feature_train)
    cleaned_data = outlierCleaner( predictions, feature_train, target_train )
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"







### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    features, target, errors = zip(*cleaned_data)
    features       = numpy.reshape( numpy.array(features), (len(features), 1))
    target = numpy.reshape( numpy.array(target), (len(target), 1))

    ### refit your cleaned data!
    try:
        reg.fit(features, target)
        print reg.score(features, target), reg.coef_
        plt.plot(features, reg.predict(features), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(features, target)
    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.show()


else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"