from sklearn import preprocessing

def transform(dataframe):
    df = preprocessing.scale(dataframe)
    return df

