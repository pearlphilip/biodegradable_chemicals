"""
Construct a neural network model from a data frame
"""

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
    df_train, df_test = train_test_split(df)#, TEST_SIZE)

    # Remove the classification column from the dataframe
    x_train = df_train.copy()
    x_train.drop(class_column_name, axis=1, inplace=True)
    x_train = x_train.values
    y_train = df_train[class_column_name].values
    y_train = y_train.astype(np.int32)

    # Create classification model
    net = NeuralNet(layers = [('input', InputLayer),
			      ('hidden0', DenseLayer),
			      ('hidden1', DenseLayer),
			      ('output', DenseLayer)],
		    input_shape = (None, x_train.shape[1]),
                    hidden0_num_units = NODES,
                    hidden0_nonlinearity = nonlinearities.sigmoid,
                    hidden1_num_units = NODES,
                    hidden1_nonlinearity = nonlinearities.sigmoid,
                    output_num_units = len(np.unique(y_train)),
                    output_nonlinearity = nonlinearities.linear,
                    update_learning_rate = 0.5,
                    verbose = 1,
                    max_epochs = 1000)

    param_grid = {'hidden0_num_units':[4, 17, 25, 32]}
    grid_search = GridSearchCV(net, param_grid, verbose=0)
    grid_search.fit(x_train,y_train)

    # l = InputLayer(shape=(None, x.shape[1]))

    # l = DenseLayer(l, num_units=NODES, nonlinearity=nonlinearities.softmax)
    # l = DropoutLayer(l, p=.2)
    # l = DenseLayer(l, num_units=NODES, nonlinearity=nonlinearities.softmax)

    #l = DenseLayer(l, num_units=len(np.unique(y)),
     #              nonlinearity=nonlinearities.softmax)
    #net = NeuralNet(l, update_learning_rate=0.5, verbose=1,
    #                max_epochs=100000)
    #net.fit(x_train, y_train)
    # print(net.score(x_train, y_train))
