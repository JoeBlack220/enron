#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print 'Number of people in the Enron dataset: {0}'.format(len(enron_data))
print 'Number of features for each person in the Enron dataset:{0}'.format(len(enron_data.values()[0]))
pois = [x for x, y in enron_data.items() if y['poi']]
print 'Number of POI\'s: {0}'.format(len(pois))
print enron_data['PRENTICE JAMES']['total_stock_value']
names = ['SKILLING JEFFREY K', 'FASTOW ANDREW S', 'LAY KENNETH L']
names_payments = {name:enron_data[name]['total_payments'] for name in names}
print sorted(names_payments.items(), key= lambda  x:x[1], reverse=True)
import pandas as pd
df = pd.DataFrame(enron_data)
quantified = sum(df.loc['salary',:] != 'NaN')
print 'number of people who have a quantifield salary {0}'.format(quantified)
isnan = sum(df.loc['total_payments',:]== 'NaN')
_,cols = df.shape
print 'total_payments == NaN : {0}, percentage = {1:.2f}%'.format(isnan, 100.*isnan/cols)
