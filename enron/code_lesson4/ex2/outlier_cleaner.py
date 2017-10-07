#!/usr/bin/python
import numpy


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    ### your code goes here
  
    cleaned_data = []
    errors = net_worths - predictions
    threshold = numpy.percentile(numpy.absolute(errors), 80)

    cleaned_data = [(ages,net_worths, errors ) for ages, net_worths, errors in zip(ages, net_worths, errors) if abs(errors) <= threshold]
    
    return cleaned_data

