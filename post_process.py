#!/usr/bin/env python

"""
Load a neural network model from a data frame
"""

import pickle

import numpy as np
import pandas as pd
from lasagne import nonlinearities
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

PICKLE = 'data.pkl'

with open(PICKLE, 'rb') as file:
    x_train = pickle.load(file)
    y_train = pickle.load(file)
    df_test = pickle.load(file)
    grid_search = pickle.load(file)
    net = pickle.load(file)

print(grid_search.grid_scores_)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)
net.save_params_to('/tmp/net.params')
