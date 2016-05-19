"""
Construct a neural network model from a data frame
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

NODES = 10
# TEST_SIZE = 0.2
PICKLE = 'data.pkl'

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
    df_train, df_test = train_test_split(df, TEST_SIZE)
    df_train, df_val = df_train[:(0.75 * len(df_train.index)), :], df_train[(0.75 * len(df_train.index)):, :]
    x_train, x_val, x_test = df_train, df_val, df_test

    # Remove the classification column from the dataframe
    x_train = x_train.drop(class_column_name, axis=1, inplace=True).values
    x_val = x_val.drop(class_column_name, axis=1, inplace=True).values
    x_test = x_test.drop(class_column_name, axis=1, inplace=True).values
    y_train = df_train[class_column_name].values.astype(np.int32)
    y_val = df_val[class_column_name].values.astype(np.int32)
    y_test = df_test[class_column_name].values.astype(np.int32)

    # Create classification model
    net = NeuralNet(layers=[('input', InputLayer),
                            ('hidden0', DenseLayer),
                            ('hidden1', DenseLayer),
                            ('output', DenseLayer)],
                    input_shape=(None, x_train.shape[1]),
                    hidden0_num_units=NODES,
                    hidden0_nonlinearity=nonlinearities.softmax,
                    hidden1_num_units=NODES,
                    hidden1_nonlinearity=nonlinearities.softmax,
                    output_num_units=len(np.unique(y_train)),
                    output_nonlinearity=nonlinearities.softmax,
                    update_learning_rate=0.1,
                    verbose=1,
                    max_epochs=100)

    param_grid = {'hidden0_num_units': [4, 17, 25],
                  'hidden0_nonlinearity': 
                  [nonlinearities.sigmoid, nonlinearities.softmax],
                  'hidden1_num_units': [4, 17, 25],
                  'hidden1_nonlinearity': 
                  [nonlinearities.sigmoid, nonlinearities.softmax],
                  'update_learning_rate': [0.01, 0.1, 0.5]}
    grid_search = GridSearchCV(net, param_grid, verbose=0)
    grid_search.fit(x_train, y_train)

    net.fit(x_train, y_train)
    print(net.score(x_train, y_train))

    with open(PICKLE, 'wb') as file:
        pickle.dump(x_train, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_train, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(df_test, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(grid_search, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(net, file, pickle.HIGHEST_PROTOCOL)
