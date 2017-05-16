# psychoPYTHON
automatizando escolha de modelos

#run.py
Contem o codigo necessario para receber argumentos e rodar o treinamento ou predicao de modelos do scikit learn ou de deep learning.

#models.py
Especificacao dos modelos de scikit learn ou deep learning.

#predict.py
Metodos para predicao do scikit learn ou deep learning.

#utils.py
Metodos utilitarios, atualmente existe um metodo para plotar a curva de aprendizagem.

#process_data.py
Metodos para leitura de dados para treinamento, teste e predição e se necessário deve ser especificado os métodos
para processamento, limpeza e estruturação dos dados.

#Exemplos de execucao de varios metodos de treinamento e predicao de modelos treinados:

##treinamento de um regressor automatico
python run.py -m autoscikit -t --prediction quantity

##treinamento de um classificador automatico
python run.py -m autoscikit -t --prediction category --labeled_data

##treinamento de um clusterizador automatico
python run.py -m autoscikit -t --prediction category --category_data

##predicao de um modelo treinado
python run.py -m GradientBoostingClassifier.pkl -p

##treinamento de um modelo deep learning
python run.py -d 1net -t

##predicao de um modelo deep learning treinado
python run.py -d deep1net.h5 -p

