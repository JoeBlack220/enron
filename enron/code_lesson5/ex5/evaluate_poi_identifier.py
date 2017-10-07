#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print confusion_matrix(labels_test, pred)

print precision_score(labels_test, pred)
print classification_report(labels_test, pred)
print recall_score(labels_test,pred)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

cm = confusion_matrix(true_labels, predictions)

print cm, '\n'
print '{0} True positives'.format(cm[1][1])
print '{0} True negatives'.format(cm[0][0])
print '{0} True positives'.format(cm[1][1])
print '{0} True positives'.format(cm[1][1])

print precision_score(true_labels, predictions)
print recall_score(true_labels, predictions)