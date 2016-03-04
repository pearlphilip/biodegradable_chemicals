import pandas as pd
from sklearn import preprocessing


def transform(dataframe, class_column_name):
    """
    Function to read dataframe and standardize the dataframe with
    a mean 0 and unit variance on every column except class_column_name.
    Converts values in column with class_column name to numeric values
    of 0 to n_classes-1.

    Parameters:
        dataframe : Input pandas dataframe
        class_column_name : Identity of the column in df with class data
    Input types: (pd.Dataframe, str)
    Output types: pd.Dataframe

    """
    cols = [col for col in dataframe.columns if col not in
            [class_column_name]]
    df = pd.DataFrame(preprocessing.scale(dataframe[cols]))

    le = preprocessing.LabelEncoder()
    le.fit(dataframe[class_column_name])
    df[class_column_name] = le.transform(dataframe[class_column_name])

    df.columns = dataframe.columns
    print(df.head())
    return df
