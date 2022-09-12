# Rede Produtora de Texto = ReProduTex
# Por Alexandre Menezes Barroso 2018-2022

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import load_model
import numpy as np
import random
import sys

np.seterr(divide = 'ignore') 

rede_neural = load_model('rede_neural.h5')

with open('corpus_completo.txt', 'r') as file:
    corpus = file.read().lower()
    
caracteres = sorted(list(set(corpus)))
limite = 40
car_indices = dict((c, i) for i, c in enumerate(caracteres))
indices_car = dict((i, c) for i, c in enumerate(caracteres))

def amostra(preds, liberdade=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / liberdade
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    
def producao_textual(tamanho, liberdade):
    inicio = random.randint(0, len(corpus) - limite - 1)
    gerado = ''
    frase = corpus[inicio: inicio + limite]
    gerado += frase
    for i in range(tamanho):
            x_pred = np.zeros((1, limite, len(caracteres)))
            for t, caractere in enumerate(frase):
                x_pred[0, t, car_indices[caractere]] = 1.

            preds = rede_neural.predict(x_pred, verbose=0)[0]
            prox_indice = amostra(preds, liberdade)
            proximo_caractere = indices_car[prox_indice]

            gerado += proximo_caractere
            frase = frase[1:] + proximo_caractere
 
            sys.stdout.write(proximo_caractere)
            sys.stdout.flush()
            
    print()

print('\n>>> Rede Produtora de Texto (ReProduTex)\n')

print(">>> Produzindo...\n")


print('-> (liberdade: 0.2)\n')
producao_textual(500, 0.4)
print('\n')

print('-> (liberdade: 0.3)\n')
producao_textual(500, 0.4)
print('\n')

print('-> (liberdade: 0.4)\n')
producao_textual(500, 0.4)
print('\n')

print('-> (liberdade: 0.5)\n')
producao_textual(500, 0.5)
print('\n')

print('-> (liberdade: 0.6)\n')
producao_textual(500, 0.6)
print('\n')


print('\n-> Encerrando programa...\n')
