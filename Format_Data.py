import numpy as np
import pandas as pd
import os
#This Function is used to format the dataset
def Read_Data():
    filename = 'biodeg.csv'
    if os.path.exists(filename):
        data = pd.read_csv(filename,sep=";")
        return data.head()
    else:
        print ('File does not exist')
        return