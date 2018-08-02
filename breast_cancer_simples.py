import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split

previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe, test_size= 0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()
classificador.add(Dense(units= 16, activation= 'relu', 
                        kernel_initializer= 'random_uniform', input_dim = 30))
classificador.add(Dense(units= 1, activation= 'sigmoid'))

classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy',
                      metrics= ['binary_accuracy'])

classificador.fit(previsores_treino, classe_treino, batch_size = 10, epochs= 100)