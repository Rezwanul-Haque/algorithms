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
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from settings.base import BASE_DIR

# dataset path
dataset_dir = BASE_DIR + '/decision_tree/dataset/student-por.csv'
## reading dataset using pandas
ds = pd.read_csv(dataset_dir, sep=';')

# print(len(ds))

## Generate binary label (pass/fail) based on G1+G2+G3(test grades, each 0-20 pts);
## threshold for passing is sum >=30
# axis=1 means use apply per row and axis=0 would mean apply per column.
ds['pass'] = ds.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis = 1)
ds = ds.drop(['G1', 'G2', 'G3'], axis=1)
# print(ds.head())  ## 5 rows * 31 columns

# use one-hot encoding on categorical columns to convert into numerical columns
ds = pd.get_dummies(ds, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                                'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
                                'internet', 'romantic'])

# print(ds.head())  ## 5 rows * 57 columns

## shuffle rows
ds = ds.sample(frac=1)

## split training and testing data
ds_train = ds[:500]
ds_test = ds[500:]

ds_train_att = ds_train.drop(['pass'], axis=1)  ## droping pass column from training data
ds_train_pass = ds_train['pass']

ds_test_att = ds_test.drop(['pass'], axis=1)    ## droping pass column from training data
ds_test_pass = ds_test['pass']

ds_att = ds.drop(['pass'], axis=1)
ds_pass = ds['pass']

## number of passing students in whole dataset:
# print("Passing: %d out of %d (%.2f%%)" %(np.sum(ds_pass), len(ds_pass), 100*float(np.sum(ds_pass)) / len(ds_pass)))

# dt = tree.DecisionTreeClassifier(criterion="entropy")
dt = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)  ## depth 5 means tree depth 5

## Build the decision tree
# dt = dt.fit(train_attributes, train_labels)
dt = dt.fit(ds_train_att, ds_train_pass)

## save tree
tree.export_graphviz(dt, out_file="result/student-performance.dot", label='all', impurity=False, proportion=True,
                    feature_names=list(ds_train_att), class_names=['fail', 'pass'], filled=True, rounded=True)

## Build the decision tree
# dt.score(test_attributes, test_labels)
score = dt.score(ds_test_att, ds_test_pass)
# print(score)

## predict a new example
# dt.predict(example_attributes)

## Average scores with cross-validation
# cross_val_score(dt, all_attributes, all_labels)
scores = cross_val_score(dt, ds_att, ds_pass, cv=5)
## show average score and +/- two standard deviations away(covering 95% 0f scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

depth_acc = np.empty((19,3), float)
i = 0

for max_depth in range(1,20):
    dt = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(dt, ds_att, ds_pass, cv=5)
    depth_acc[i, 0] = max_depth
    depth_acc[i, 1] = scores.mean()
    depth_acc[i, 2] = scores.std() * 2
    i += 1

    # print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))

# print(depth_acc)

# creating a plot to show result in a graph
fig, ax = plt.subplots()
ax.errorbar(depth_acc[:, 0], depth_acc[:, 1], yerr=depth_acc[:, 2])
plt.show()