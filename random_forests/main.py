"""Implementation of Random Forests algorithm on bird species dataset

#TODO 
1. Classification and techniques for evaluation
2. Predicting bird species with random forests
3. Confusion matrix

Random forests are extensions of decision trees and are a kind of ensemble method.
Ensemble methods can achieve high accuracy by building several classifiers and running a
each one independently.

# description from dataset README:

The files in this folder contain attribute labels obtained from Mechanical
Turk workers on the Birds-200 dataset. The file labels.txt contains the actual
labels. Each line corresponds to one attribute label of the form: 
<image_id> <attribute_id> <is_present> <certainty_id> <worker_id>

The file 'images.txt' contains lines of the form: 
<image_id> <image_file_name>

The file 'images-dirs.txt' is the same as 'images.txt' but also includes the
directory names: 
<image_id> <full_image_path>

The file 'attributes.txt' contains lines of the form: 
<attribute_id> <attribute_name>

<is_present> is 0 or 1, and indicates whether or not the worker thought the
given attribute was present in the given image.

The file 'certainties.txt' contains lines of the form: 
<certainty_id> <certainty_name>

Each unique value of worker_id corresponds to a unique worker on Mechanical
Turk.

"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree, svm
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from settings.base import BASE_DIR

## Ploting the confusion matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix
    Normalization can be applied by settings 'normalize=True'
    
    Arguments:
        cm {[type]} -- [description]
        classes {[type]} -- [description]
    
    Keyword Arguments:
        normalize {bool} -- [description] (default: {False})
        title {str} -- [description] (default: {'Confusion matrix'})
        cmap {[type]} -- [description] (default: {plt.cm.Blues})
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("confusion matrix, without normalization")

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # for i, j in itertools.product(range(cm.shape[0], range(cm.shape[1]))):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalignment="center",
    #             color="white" if cm[i, j] > thresh else "black"
    #     )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')





# Some lines have too many fields (?), so skip bad lines
imgatt = pd.read_csv(BASE_DIR + '/random_forests/dataset/CUB_200_2011/attributes/image_attribute_labels.txt', sep='\s+', header=None,
                    error_bad_lines=False, warn_bad_lines=False, usecols=[0,1,2], names=['imgid', 'attid', 'present'])

# print(imgatt.head())

# print(imgatt.shape)

# Need to reorganize imgatt to have one row per imgid, and 312 columns (one column per attribute), 
# with 1/0 in each cell representing if that imgid has that attribute or not

imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present')

# print(imgatt2.head())  ## 5 rows * 312 columns

# print(imgatt2.shape)

# Now we need to load the image true classes
imglabels = pd.read_csv(BASE_DIR + '/random_forests/dataset/CUB_200_2011/image_class_labels.txt', sep=' ',
                        header=None, names=['imgid', 'label'])

imglabels = imglabels.set_index('imgid')

# print(imglabels.head())

# print(imglabels.shape)

# Now we need to attach the labels to the attribute data set, and shuffle; then we'll separate a test set from a traning set

df = imgatt2.join(imglabels)
df = df.sample(frac=1)

df_att = df.iloc[:, :312]
df_label = df.iloc[:, 312:]

# print(df_att.head())  # 5 rows * 312 columns

df_train_att = df_att[:8000]
df_train_label = df_label[8000:]

df_test_att = df_att[:8000]
df_test_label = df_label[8000:]

df_train_label = df_train_label['label']
df_test_label = df_test_label['label']

## Implementing Random forest
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)

clf.fit(df_train_att, df_train_label)

# print(clf.predict(df_train_att.head()))

# print(clf.score(df_test_att, df_test_label))

## Creating a confusion matrix

predict_labels = clf.predict(df_test_att)
cm = confusion_matrix(df_test_label, predict_labels)

# print(cm)

birds = pd.read_csv(BASE_DIR + '/random_forests/dataset/CUB_200_2011/classes.txt',
                    sep='\s+', heade=None, usecols=[1], names=['birdname'])

birds = birds['birdname']

# print(birds)

np.set_printoptions(precision=2)
plt.figure(figsize=(60,60), dpi=300)
plot_confusion_matrix(cm, classes=birds, normalize=True)
plt.show()

# Comparaing results with Decision tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(df_train_att, df_train_label)
# print(clftree.score(df_train_att, df_train_label))

# Comparaing results with SVM
clfsvm = svm.SVC()
clfsvm.fit(df_train_att, df_train_label)
# print(clfsvm.score(df_train_att, df_train_label))

## The random forest is still better.

# Cross validation check
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
## show average score and +/- two standard deviations away(covering 95% 0f scores)
print("Accuracy of RF: %0.2f (+/- %0.2f)" %(scores.mean(), scores.std() * 2))

scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5)
## show average score and +/- two standard deviations away(covering 95% 0f scores)
print("Accuracy of DT: %0.2f (+/- %0.2f)" %(scorestree.mean(), scorestree.std() * 2))


scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5)
## show average score and +/- two standard deviations away(covering 95% 0f scores)
print("Accuracy of SVM: %0.2f (+/- %0.2f)" %(scoressvm.mean(), scoressvm.std() * 2))


max_features_opts = range(5, 50, 5)
n_estimators_opts = range(10, 200, 20)
rf_params = np.empty((len(max_features_opts) * len(n_estimators_opts), 4), float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)

        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i, 3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" % \
              (max_features, n_estimators, scores.mean(), scores.std() * 2))

fig = plt.figure()
fig.clf()
ax = fig.gca(projection='3d')
x = rf_params[:, 0]
y = rf_params[:, 1]
z = rf_params[:, 2]

ax.scatter(x, y, z)
ax.set_zlim(0.2, 0.5)
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')

plt.show()

