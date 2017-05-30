from sklearn import datasets
import numpy as np 
import pandas as pd

def load_processed_data_test():
    df = pd.read_csv("C:\\Users\\rafaela.oliveira\\Desktop\\base_waze.csv", sep=',')
    Y = df['totaldias']
    
    X = df[['cdTipoCartorio','cdForo','cdVara','cdClasseExt','qtParteAtiva','qtPartePassiva','vlCausa']]
    X['cdTipoCartorio'] = X['cdTipoCartorio'].astype('category')
    X['cdForo'] = X['cdForo'].astype('category')
    X['cdVara'] = X['cdVara'].astype('category')
    X['cdClasseExt'] = X['cdClasseExt'].astype('category')
    return X, Y
load_processed_data_test()

def load_predict_data_test():
    df = pd.read_csv("C:\\Users\\rafaela.oliveira\\Desktop\\base_waze.csv", sep=',')
    Y = df['totaldias']
    X = df[['cdTipoCartorio','cdForo','cdVara','cdClasseExt','qtParteAtiva','qtPartePassiva','vlCausa']]
    return X
