import pandas as pd
from sklearn import preprocessing


def transform(dataframe, class_column_name):
    cols = [col for col in dataframe.columns if col not in [class_column_name]]
    df = pd.DataFrame(preprocessing.scale(dataframe[cols]))
    
    le = preprocessing.LabelEncoder()
    le.fit(dataframe[class_column_name])
    df[class_column_name] = le.transform(dataframe[class_column_name])

    df.columns = dataframe.columns
    print(df.head())
    return df

