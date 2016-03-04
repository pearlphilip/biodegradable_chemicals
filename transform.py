import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

TEST_SIZE = 0.2


def transform(dataframe, class_column_name):
    """
    Function to read dataframe and standardize the dataframe with
    a mean 0 and unit variance on every column except class_column_name.
    Converts values in column with class_column name to numeric values
    of 0 to n_classes-1.
    Splits dataframe into training and test sets in a given ratio.
    Returns list of training and test dataframes.

    Input_args: (dataframe, class_column_name)
    Input types: (pd.Dataframe,str)
    Output types:(pd.Dataframe, pd.Dataframe)

    """
    cols = [col for col in dataframe.columns if col not in
            [class_column_name]]
    df = pd.DataFrame(preprocessing.scale(dataframe[cols]))

    le = preprocessing.LabelEncoder()
    le.fit(dataframe[class_column_name])
    df[class_column_name] = le.transform(dataframe[class_column_name])

    df.columns = dataframe.columns
    df_train, df_test = train_test_split(df, TEST_SIZE)
    print(df_train.head(), df_test.head())
    return df_train, df_test
