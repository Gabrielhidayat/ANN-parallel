#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 02:15:20 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import time

start=time.time()

df = pd.read_csv('/home/gabriel/TUGAS/AI/Tugas Kelompok 2/TUGAS KELOMPOK 3/heart.csv')

X=df.drop(columns = 'output')
y= df['output']

sc = StandardScaler()
x=sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense

aktivasi = []
jumlah_layer = []
jumlah_nodes = []
error = []
akurasi = []
def model_ANN(activation,layer,nodes):
    
    model = Sequential()

    model.add(Dense(
            units = nodes,
            kernel_initializer="uniform",
            activation=activation,
            input_dim = X.shape[1]
            ))
   
    for _ in range (3):
        model.add(Dense(
            units=nodes,
            kernel_initializer="uniform",
            activation=activation,
            ))
    model.add(Dense(
            units = 1,
            ))
    
    model.compile(
            optimizer = "adam",
            loss="mse",
            metrics=['accuracy']
            )
    
    simulasi=model.fit(
            X_train,
            Y_train,
            batch_size=10,
            epochs=10,
            verbose=0
            )
    
    
    print(f"activation = {activation}, jumlah layer = {layer}, jumlah node = {nodes}")
    
    akurasi_epoch=(simulasi.history)
    
    print('akurasi traning=',akurasi_epoch['accuracy'][-1])
    
    hasil=model.evaluate(x,y)[1]
    loss=model.evaluate(x,y)[0]
    print('hasil akhir = ',hasil)
    print("\n")
    
    aktivasi.append(activation)
    jumlah_layer.append(layer)
    jumlah_nodes.append(nodes)
    error.append(loss)
    akurasi.append(hasil)
    
activations=['relu','sigmoid','tanh','linear']
layers=[1,2,3]
nodes=[4,8,12,16,20,24,32,64]



for node in nodes:
    for layer in layers:
        for activation in activations:
            
            model_ANN(activation, layer, node)

end=time.time()-start
print('runtime=',end)

data_akurasi = pd.DataFrame(aktivasi).rename(columns={0:'aktivasi'})
data_akurasi['jumlah layer'] = jumlah_layer
data_akurasi['jumlah nodes'] = jumlah_nodes
data_akurasi['error']        = error
data_akurasi['akurasi']      = akurasi

best_akurasi=data_akurasi.loc[data_akurasi['akurasi'].idxmax()]
print('\nnilai akurasi terbaik adalah\n',best_akurasi)

data_akurasi.to_csv('/home/gabriel/TUGAS/HPC/Pak Dzaki/RBL/ANN serial.csv',index=False)
            


