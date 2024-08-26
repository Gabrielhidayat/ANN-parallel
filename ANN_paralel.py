#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 07:09:16 2023

@author: gabriel
"""

from mpi4py import MPI
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import time

start=time.time()

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

size = comm.Get_size()

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
            batch_size=30,
            epochs=300,
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

local_n = len(nodes) // size

local_nodes = [nodes[i*local_n:i+local_n] if i < (size-1) else nodes[i*local_n:] for i in range(size)]

nodes = comm.scatter(local_nodes, root=0)

for node in nodes:
    for layer in layers:
        for activation in activations:
            model_ANN(activation, layer, node)
            

end=time.time()-start
print('core = ',size)
print('runtime=',end)



if rank==0:
    aktivasi = comm.gather(aktivasi,root=0)
    jumlah_layer = comm.gather(jumlah_layer,root=0)
    jumlah_nodes = comm.gather(jumlah_nodes,root=0)
    error = comm.gather(error,root=0)
    akurasi = comm.gather(akurasi,root=0)
    
    aktivasi=np.asarray(aktivasi)
    jumlah_layer=np.asarray(jumlah_layer)
    jumlah_nodes=np.asarray(jumlah_nodes)
    error=np.asarray(error)
    akuras=np.asarray(akurasi)

    data_akurasi = pd.DataFrame(aktivasi.reshape(-1,1)).rename(columns={0:'aktivasi'})
    data_akurasi['jumlah layer'] = jumlah_layer.reshape(-1,1)
    data_akurasi['jumlah nodes'] = jumlah_nodes.reshape(-1,1)
    data_akurasi['error']        = error.reshape(-1,1)
    data_akurasi['akurasi']      = akurasi.reshape(-1,1)
    
    print('tipe data :',type(data_akurasi))
    
    best_akurasi=data_akurasi.loc[data_akurasi['akurasi'].idxmax()]
    print('\nnilai akurasi terbaik adalah\n',best_akurasi)
    
    data_akurasi.to_csv('/home/gabriel/TUGAS/HPC/Pak Dzaki/RBL/ANN paralel core '+str(size)+'.csv',index=False)
    