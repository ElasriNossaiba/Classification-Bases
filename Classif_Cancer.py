#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nossaiba
"""

# import required modules
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt


# Load Dataset
data_set = datasets.load_breast_cancer()
X=data_set.data
y=data_set.target

# entrés
print ('\nInput features:')
print (data_set.feature_names)

# sorties
print ('\nTarget:')
print (data_set.target_names)

# Base d'entrainement et test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)

# model regression logistique  de sklearn
model=LogisticRegression()
model.fit(X_train,y_train)

#prediction
y_pred=model.predict(X_test)
#performance
correct = (y_test == y_pred).sum()
incorrect = (y_test != y_pred).sum()
accuracy = correct / (correct + incorrect) * 100

print('\nPercent Accuracy: %0.1f' %accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('\nMatrice de confusion :' )
print(cm)

