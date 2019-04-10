"""Implementing Decision Tree Algorithm
## Author: Rezwanul Haque

# Q. When should we use decision trees?
# A. Decision trees can perform better when the data has few attributes, but may perform poorly
when the data has many attributes.
# Explain: This is because the tree may grow too large to be understandable and could easily overfit
the training data by introducing branches that are too specific to the training data and don't
really bear any relation to the test data created, this can reduce the chance of getting an 
accurate result.

## Dataset: UC Irvine Student Performance Dataset
# Deatil: 
    Attributes: 30
    Student: 649
"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.

from sklearn import tree
import pandas as pd

# from settings import *
from settings.base import BASE_DIR

dataset_dir = BASE_DIR + '/decision_tree/dataset/student-por.csv'

ds = pd.read_csv(dataset_dir, sep=';')

print(len(ds))

# dt = tree.DecisionTreeClassifier(criterion="entropy")

## Build the decision tree
# dt = dt.fit(train_attributes, train_labels)
## Build the decision tree
# dt.score(test_attributes, test_labels)
## predict a new example
# dt.predict(example_attributes)
## Average scores with cross-validation
# cross_val_score(dt, all_attributes, all_labels)
