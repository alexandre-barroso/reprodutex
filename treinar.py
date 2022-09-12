#ReProduTex
#Por Alexandre Menezes Barroso 2018-2022

#deleta conteúdo do corpus antigo
corpus_zerado = open('corpus_completo.txt', 'r+')
corpus_zerado.truncate(0)
corpus_zerado.close()

#abre corpus parcial 1
with open('corpus_1.txt', 'r') as file :
  corpus_1 = file.read()

#abre corpus parcial 2
with open('corpus_2.txt', 'r') as file:
  corpus_2 = file.read()
  
  #transforma em string
  corpus_1_txt = str(corpus_1)
  corpus_2_txt = str(corpus_2)
  
  #soma ambos corpora
  corpus_completo = corpus_1_txt + corpus_2_txt

#escreve as strings no arquivo vazio
with open('corpus_completo.txt', 'w') as file:
  file.write(corpus_completo)
  
#abre corpus
with open('corpus_completo.txt', 'r') as file :
  corpus = file.read()

#mudança no corpus
### INSERIR AQUI TODAS AS MUDANÇAS
#corpus = corpus.replace("e'", 'é')

#reescreve corpus
with open('corpus_completo.txt', 'w') as file:
  file.write(corpus)
  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import random
import sys

from keras.models import load_model
from keras.callbacks import *
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

print('\n>>> Inicializando corpus...\n')

#Manipulação textual:

with open('corpus_completo.txt', 'r') as file:
    corpus = file.read().lower()

print('Tamanho das amostras:', len(corpus))

caracteres = sorted(list(set(corpus))) #coletando todos os caracteres únicos

print('Quantidade de caracteres únicos: ',len(caracteres))

car_indices = dict((c, i) for i, c in enumerate(caracteres))

indices_car = dict((i, c) for i, c in enumerate(caracteres))

limite = 40

passos = 3

frases = []

proximos_caracteres = []

for i in range(0, len(corpus) - limite, passos):
    frases.append(corpus[i: i + limite])
    proximos_caracteres.append(corpus[i + limite])
    
x = np.zeros((len(frases), limite, len(caracteres)), dtype=bool)

y = np.zeros((len(frases), len(caracteres)), dtype=bool)

for i, frase in enumerate(frases):
    for t, caractere in enumerate(frase):
        x[i, t, car_indices[caractere]] = 1
    y[i, car_indices[proximos_caracteres[i]]] = 1

### FASE DE TREINO

print('\n>>> Inicializando rede neural...\n')
  
#Rede neural:
#1. Camada LSTM
#2. Camada densa totalmente conectada
#3. Função de ativação softmax
#RMSprop + Categorical Crossentropy

rede_neural = Sequential()
rede_neural.add(LSTM(128, input_shape=(limite, len(caracteres))))
rede_neural.add(Dense(len(caracteres)))
rede_neural.add(Activation('softmax'))
    
print('>>> DIGITE: \n\n 1. "c" para continuar treino de onde parou. \n\n 2. "r" para re-treinar do começo. \n\n 3. qualquer outra coisa para encerrar o programa. \n\n OBSERVAÇÃO: digite "c" ou "r" sem aspas!\n')

opcao = input('> ')

print('')

if opcao.lower() == 'r':

    epoch = input('\n> Quantas iterações (epochs) a rede neural deve rodar?\n>  ')
    epoch = int(epoch)
    batch_size = input('\n> Qual o batch_size?\n>  ')
    batch_size = int(batch_size)
    otimizador = RMSprop(learning_rate=0.01)
    rede_neural.compile(loss='categorical_crossentropy', optimizer=otimizador)
    peso_salvos = "neuronios.hdf5"
    checkpoint = ModelCheckpoint(peso_salvos, monitor='loss',
                                 verbose=1, save_best_only=True,
                                 mode='min')
    redutor = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=1, min_lr=0.001)
    callbacks = [checkpoint, redutor]
    rede_neural.fit(x, y, batch_size=batch_size, epochs=epoch, callbacks=callbacks)
    rede_neural.save('rede_neural.h5')

    print('\n>>> Finalizando treino da rede neural...\n')
    
elif opcao.lower() == 'c':

    epoch = input('\n> Quantas iterações (epochs) a rede neural deve rodar?\n>  ')
    epoch = int(epoch)
    batch_size = input('\n> Qual o batch_size?\n>  ')
    batch_size = int(batch_size)
    del rede_neural
    rede_neural = load_model('rede_neural.h5')
    peso_salvos = "neuronios.hdf5"
    checkpoint = ModelCheckpoint(peso_salvos, monitor='loss',
                                 verbose=1, save_best_only=True,
                                 mode='min')
    redutor = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=1, min_lr=0.001)
    callbacks = [checkpoint, redutor]
    rede_neural.fit(x, y, batch_size=batch_size, epochs=epoch, callbacks=callbacks)
    rede_neural.save('rede_neural.h5')

    print('\n>>> Finalizando treino da rede neural...\n')

else:

   print('\n>>> Encerrando programa...\n')    
