from sklearn import datasets
import numpy as np
import pandas as pd


#import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def load_processed_data_test():
    df = pd.read_csv("data", sep=';')
    X = df[['variaveis explicativas']]
    y = pd.to_numeric(df['resposta'])


# tratando variáveis categóricas
    category_map = {}
    category_list = ['VARIAVEIS CATEGORIAS']

    # for each category transform into numbers
    for cat in category_list:
        encoder = LabelEncoder()
        X[cat] = encoder.fit_transform(X[cat]) # fitting this category and trasnforming the column to indexes
        category_map[cat] = encoder.classes_   # saving the indexes to know how to go back from index->category

# gerando matriz de OneHotEncoder
    # gerando matriz de OneHotEncoder
    categorical_indexes=[]
    for cat in category_list:
        categorical_indexes.append(X.columns.get_loc(cat))
    one_hot_encoder = OneHotEncoder(categorical_features = categorical_indexes)
    print(X.head())
    print(y.head())
    return X, y
