#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nossaiba
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0) 
m=16

#Dataset
x1=np.array([5,3,4,5,0,8,10,7,15,17,16,11,14,14,12,18]).reshape(m,1)
x2=np.array([2,5,8,11,7,10,12,10,11,14,14,14,16,12,12,15]).reshape(m,1)
y =np.array([0, 0, 0,0,0,0, 0, 1, 1, 1, 1,1,1, 1,1, 1]).reshape(m,1)

fig, ax = plt.subplots()
plt.scatter(x1[0:7], x2[0:7], c='r', marker='x', label='Not Admitted')
plt.scatter(x1[8:16], x2[8:16], c='b', marker='o', label='Admitted')
plt.legend()
ax.set_xlabel('Exam 1 ')
ax.set_ylabel('Exam 2 ')
plt.show()

#variable
X = np.hstack((x1, x2, np.ones(x1.shape)))
theta = np.random.randn(3, 1)

#==============================================================================
#model
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#nums = np.arange(-10, 10, step=1)
#fig, ax = plt.subplots()
#ax.plot(nums, sigmoid(nums), 'b')

def erreur(theta, X, y):
    m = len(y)
    first = np.multiply(-y, np.log(sigmoid(X.dot(theta))))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X.dot(theta))))
    return np.sum(first - second) / m

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(sigmoid(X.dot(theta)) - y)

#desente du gradient 
def gradient_descent(X, y, theta, alpha, n_iterations):
    err_history = np.zeros(n_iterations) 
    for i in range(0, n_iterations):
        theta = theta - alpha * grad(X, y, theta) 
        err_history[i] = erreur(theta,X,y)         
    return theta, err_history

#==============================================================================
n_iterations = 1000
alpha = 0.09

theta_optimal, err_history = gradient_descent(X, y, theta, alpha, n_iterations)


def predict(theta, X):
    probability = sigmoid(X.dot(theta))
    return [1 if x >= 0.5 else 0 for x in probability]

predictions = predict(theta_optimal, X)

#=============================================================================
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in  zip(predictions, y)]

accuracy = (sum(correct)/len(correct))*100
print ('accuracy = {0}%'.format(accuracy))


#matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, predictions)
print('matrice de confusion')
print(cm)
#
#
#courbe ROC
from sklearn.metrics import roc_curve
y_pred_proba= sigmoid(X.dot(theta_optimal))
fpr,tpr,thresholds=roc_curve(y,y_pred_proba)
plt.plot([0, 1],[0, 1],'--')
plt.plot(fpr,tpr,label='linear regression')
plt.xlabel('false positive')
plt.ylabel('false negative')
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted not-Admitted', 'Predicted Admitted'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual not-Admitted', 'Actual Admitted'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


