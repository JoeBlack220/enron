#!/usr/bin/python

import sys
import pickle
import pandas as pd
sys.path.append("../tools/")
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features
features_list_all = ['salary',"bonus","to_messages",'total_payments','exercised_stock_options',
                 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees',
                 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
features_list = ["poi"]


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for item, value in data_dict.items():
    if value['exercised_stock_options']=='NaN' or value['bonus']== 'NaN' or value['salary']=='NaN':
        value['fin1'] = 'NaN'
    else:
        value["fin1"] = value['exercised_stock_options']/float(value['bonus']+value['salary'])

    if value['to_messages'] == 'NaN' or value['from_poi_to_this_person'] == 'NaN':
        value['to_fraction'] = 'NaN'
    else:
        value['to_fraction'] = value['from_poi_to_this_person']/float(value['to_messages'])

    if value['from_messages'] == 'NaN' or value['from_this_person_to_poi'] == 'NaN':
        value['from_fraction'] = 'NaN'
    else:
        value['from_fraction'] = value['from_this_person_to_poi']/float(value['from_messages'])


features_list_all.append('fin1')
features_list_all.append('to_fraction')
features_list_all.append('from_fraction')
features_list += features_list_all
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

cv = StratifiedShuffleSplit(labels,n_iter = 100, test_size=0.4,random_state=45)
steps = [('skb', SelectKBest()), ('nb', GaussianNB())]
clf = Pipeline(steps)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
param_grid = {'skb__k':range(1,len(features_list_all))}
grid = GridSearchCV(clf, param_grid ,verbose=True, cv=cv, scoring = 'f1' )
grid.fit(features,labels)
print "best estimator:", grid.best_estimator_
print "best F1:", grid.best_score_
clf = grid.best_estimator_

selector = SelectKBest(f_classif, k=5)
selector.fit(features, labels)
features_df = pd.DataFrame({'feature': features_list_all,
                           'score': selector.scores_})
features_df.sort_values(by="score", ascending=False, inplace=True)
best_features = list(features_df["feature"][0:5])
print best_features

features_list = ["poi"] + best_features
print features_list


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)