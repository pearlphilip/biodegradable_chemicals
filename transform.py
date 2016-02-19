from sklearn import preprocessing

def transform(dataframe):
    df = preprocessing.scale(dataframe[:-1])
    return df

