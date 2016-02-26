"""
Construct a neural network model from a data frame
"""

import numbers

import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
import numpy as np
import pandas as pd

NODES = 10


def build_nn(df=None, class_column_name=None):
    """
    Construct a classification neural network model from input dataframe
    
    Parameters:
        df : input dataframe
        class_column_name : identity of the column in df with class data
    """

    # Type check inputs for sanity
    if df is None:
        raise ValueError('df is None')
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a dataframe')
    if class_column_name is None:
        raise ValueError('class_column_name is None')
    if not isinstance(class_column_name, basestring):
        raise TypeError('class_column_name is not a string')
    if class_column_name not in df.columns:
        raise ValueError('class_column_name (%s) is not a valid column name'
                         % class_column_name)

    df = df.sample(frac=1).reset_index(drop=True)

    # Remove the classification column from the dataframe
    x = df.copy()
    x.drop(class_column_name, axis=1, inplace=True)
    x = x.values
    y = df[class_column_name].values
    y = y.astype(np.int32)

    # Create classification model
    l = InputLayer(shape=(None, x.shape[1]))

    l = DenseLayer(l, num_units=NODES, nonlinearity=nonlinearities.softmax)
    # l = DropoutLayer(l, p=.2)
    # l = DenseLayer(l, num_units=NODES, nonlinearity=nonlinearities.softmax)

    l = DenseLayer(l, num_units=len(np.unique(y)),
                   nonlinearity=nonlinearities.softmax)
    net = NeuralNet(l, update_learning_rate=0.5, verbose=1,
                    max_epochs=100000)
    net.fit(x, y)
    print(net.score(x, y))
