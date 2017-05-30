from sklearn import datasets
import numpy as np
import pandas as pd



def load_processed_data_test():
    df = pd.read_csv("data", sep=';')
    df = df[df['STATUS_PROCESSO']=='BAIXADO']

    Y = df['totaldias'].astype(int)
    X = df[['Segmento', 'flArea', 'flProcVirtual', 'cdTipoCartorio', 'cdClasseExt','cdAssuntoExt','cdComarca','cdForo','cdVara','cdRaj','nuEntrancia','qtParteAtiva','qtPartePassiva','qtTestemunha', 'qtAudiencia','TemRecurso', 'flReuPreso', 'flPrioridadeIdoso', 'qtMovAposBaixa', 'flJusticaGratuita', 'qtMov', 'TeveAudiencia', 'TeveAcordo', 'qtDocEmitido', 'TemDefensoria', 'TemPerito',  
    'tpPessoaAtiva', 'tpGovernoAtiva', 'tpPessoaPassiva', 'tpGovernoPassiva', 'cdAgente', 'qtMovMag', 'qtDocEmitidoMag', 'totaldias', 'vlCausa', 'idade_passiva', 'idade_ativa']]
   

    X['cdTipoCartorio'] = X['cdTipoCartorio'].astype(int).astype('category')
    X['cdForo'] = X['cdForo'].astype(int).astype('category')
    X['cdVara'] = X['cdVara'].astype(int).astype('category')
    X['cdClasseExt'] = X['cdClasseExt'].astype(int).astype('category')
    X['Segmento'] = X['Segmento'].astype('category')
    X['flArea'] = X['flArea'].astype('category')
    X['flProcVirtual'] = X['flProcVirtual'].astype('category')
    X['cdTipoCartorio'] = X['cdTipoCartorio'].astype(int).astype('category')
    X['cdClasseExt'] = X['cdClasseExt'].astype(int).astype('category')
    X['cdComarca'] = X['cdComarca'].astype(int).astype('category')
    X['cdRaj'] = X['cdRaj'].astype(int).astype('category')
    X['nuEntrancia'] = X['nuEntrancia'].astype(int).astype('category')
    X['flJusticaGratuita'] = X['flJusticaGratuita'].astype('category')
    X['TeveAudiencia'] = X['TeveAudiencia'].astype('category')
    X['TeveAcordo'] = X['TeveAcordo'].astype('category')
    X['TemDefensoria'] = X['TemDefensoria'].astype('category')
    X['TemPerito'] = X['TemPerito'].astype('category')
    X['tpPessoaAtiva'] = X['tpPessoaAtiva'].astype('category')
    X['tpGovernoAtiva'] = X['tpGovernoAtiva'].astype('category')
    X['tpPessoaPassiva'] = X['tpPessoaPassiva'].astype('category')
    X['tpGovernoPassiva'] = X['tpGovernoPassiva'].astype('category')
    X['cdAgente'] = X['cdAgente'].astype(int).astype('category')
    X['idade_passiva'] = X['idade_passiva'].astype('category')
    X['idade_ativa'] = X['idade_ativa'].astype('category')
    X.fillna(0)
    ### transformando variaveis caracteres
    X = pd.get_dummies(X)

    #print(Y.dtypes)
    #print(X.dtypes)
    #print(X.head())
    #print(Y.head())
    return X, Y
load_processed_data_test()


#def load_predict_data_test():
#    df = pd.read_csv("C:\\Users\\rafaela.oliveira\\Desktop\\base_waze.csv", sep=',')
 #   Y = df['totaldias']
 #   X = df[['cdTipoCartorio','cdForo','cdVara','cdClasseExt','qtParteAtiva','qtPartePassiva','vlCausa']]
 #   return X
