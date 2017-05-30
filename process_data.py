from sklearn import datasets
import numpy as np 
import pandas as pd

def load_processed_data_test():
    df = pd.read_csv('~/data/dataframe_output2.csv', sep=',')
    Y = df['lifecycle_stage']
    
    X = df[['conversions_sum', 'leads_float']]
#    X = df[['job_title', 'area_de_atuacao_da_empresa', 'conversions_sum', 'leads_float']]
#    X['job_title'] = X['job_title'].astype('category')
#    X['area_de_atuacao_da_empresa'] = X['area_de_atuacao_da_empresa'].astype('category')
    return X, Y

def load_predict_data_test():
    df = pd.read_csv('~/data/dataframe_output2.csv', sep=',')
    Y = df['lifecycle_stage']
    X = df[['conversions_sum', 'leads_float']]
    #X = df[['job_title', 'area_de_atuacao_da_empresa', 'conversions_sum', 'leads_float']]
    return X

load_processed_data_test()
