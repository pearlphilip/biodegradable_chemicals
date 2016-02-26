import numpy as np
import pandas as pd
import os


def read_data(filename):
    """
    This Function is used to format the dataset
    """
    if os.path.exists(filename):
        data = pd.read_csv(filename, sep=";")
        print("Showing the first 5 lines of %s" % filename)
        print(data.head())
        return data
    else:
        raise FileNotFoundError(filename)
