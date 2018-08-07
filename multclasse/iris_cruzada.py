import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede(activation, dropout, neurons):
    classificador = Sequential()
    classificador.add(Dense(units= neurons, activation= activation, input_dim = 4))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units= 3, activation= 'softmax'))
    classificador.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn= criar_rede)

parametros = {'batch_size': [10, 12, 15],
              'neurons': [3, 4, 5],
              'dropout': [0.1, 0.2, 0.3],
              'epochs': [500, 1000, 1500],
              'activation': ['relu', 'sigmoid', 'softmax']}

grid_search = GridSearchCV(estimator= classificador, param_grid= parametros, cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_param = grid_search.best_params_
melhor_preci = grid_search.best_score_