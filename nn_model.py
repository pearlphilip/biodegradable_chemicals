'''
Construct a neural network model from a data frame
'''

import numbers

import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
import numpy as np
import pandas as pd

NODES=18

def build_nn(df=None, class_column=None):
    '''
    Construct a classification neural network model from input dataframe
    
    Parameters:
        df : input dataframe
        class_column : identity of the column in df with class data
    '''

    # Type check inputs for sanity
    if df is None:
        raise ValueError('df is None')
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a dataframe')
    if class_column is None:
        raise ValueError('class_column is None')
    if not isinstance(class_column, numbers.Integral):
        raise TypeError('class_column is not an integer')
    if df.shape()[1] < class_column:
        raise ValueError('class_column (=%d) is larger than the number of columns' % (class_column))

    # Remove the classification column from the dataframe
    x = df.copy()
    x.drop(class_column, axis=1)
    y = df[class_column]

    # Create classification model
    l = InputLayer(shape=(None, x.shape[1]))
    # Temporary - let's remove the second layer
    # l = DenseLayer(l, num_units=NODES, nonlinearity=nonlinearities.softmax)
    l = DenseLayer(l, num_units=len(np.unique(y)),
                   nonlinearity=nonlinearities.softmax)
    net = NeuralNet(l, update_learning_rate=0.5, verbose=1,
                    max_epochs=100)
    net.fit(x, y)
    print(net.score(x, y))

