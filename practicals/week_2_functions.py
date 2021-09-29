# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:23:30 2021

@author: s160518
"""

# add subfolder that contains all the function implementations
# to the system path so we can import them
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import sys
sys.path.append('code/')

diabetes = load_diabetes()

breast_cancer = load_breast_cancer()

# the actual implementation is in linear_regression.py,
# here we will just use it to fit a model
#from linear_regression import *

# load the dataset
# same as before, but now we use all features
X_train = diabetes.data[:300, :]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, :]
y_lest = diabetes.target[300:, np.newaxis]

#beta = lsq(X_train, y_train)

# print the parameters
#print(beta)