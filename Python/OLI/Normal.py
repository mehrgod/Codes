# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:06:26 2020

@author: mirza
"""

import numpy as np
from sklearn import preprocessing

path = 'C:/Project/EDU/OLI_175318/update/step/sep/tfidf/'

id = []

with open(path + 'SequenceVector.txt') as file:
    array2d = [[int(digit) for digit in line.split('\t')[1].split(',')] for line in file]

X = np.array(array2d)

X_normalized = preprocessing.normalize(X, norm='l1')

np.savetxt(path + "VectorNormal1.csv", X_normalized, delimiter=",")

file.close()