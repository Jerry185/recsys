#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:46:38 2018

@author: ben
"""

import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop, Adam

#path = "ml-20m/"
#path = 'ml-10M100K/'
path = 'ml-1m/'

ratings = pd.read_csv(path+'ratings.dat', sep='::')
movie_names = pd.read_csv(path+'movies.dat', sep='::')


usCol = '1'
mvCol = '1193'
rtCol = '5'
users = ratings[usCol].unique()
movies = ratings[mvCol].unique()


userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}


ratings[mvCol] = ratings[mvCol].apply(lambda x: movieid2idx[x])
ratings[usCol] = ratings[usCol].apply(lambda x: userid2idx[x])


user_min, user_max, movie_min, movie_max = (ratings[usCol].min(),
                                            ratings[usCol].max(), ratings[mvCol].min(), ratings[mvCol].max())

n_users = ratings[usCol].nunique()
n_movies = ratings[mvCol].nunique()



#latent factors
n_factors = 50


#split train and validation
np.random.seed = 42

msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]




'''
#####################################
Dot product model
#####################################
'''

user_in = Input(shape=(1,), dtype='int64', name='user_in')
u = Embedding(n_users, n_factors, input_length=1, W_regularizer=l2(1e-4))(user_in)
movie_in = Input(shape=(1,), dtype='int64', name='movie_in')
m = Embedding(n_movies, n_factors, input_length=1, W_regularizer=l2(1e-4))(movie_in)

x = merge([u, m], mode='dot')
x = Flatten()(x)
model = Model([user_in, movie_in], x)



#Adding bias

def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)

user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)

def create_bias(inp, n_in):
    x = Embedding(n_in, 1, input_length=1)(inp)
    return Flatten()(x)

ub = create_bias(user_in, n_users)
mb = create_bias(movie_in, n_movies)

x = merge([u, m], mode='dot')
x = Flatten()(x)
x = merge([x, ub], mode='sum')
x = merge([x, mb], mode='sum')
model = Model([user_in, movie_in], x)


model.compile(Adam(0.001), loss='mse')



model.fit([trn[usCol], trn[mvCol]], trn[rtCol], batch_size=64, nb_epoch=6,
          validation_data=([val[usCol], val[mvCol]], val[rtCol]))




'''
    #####################################
    Neural network model
    #####################################
'''

user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)

x = merge([u, m], mode='concat')
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(70, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(1)(x)
nn = Model([user_in, movie_in], x)
nn.compile(Adam(0.001), loss='mse')



nn.fit([trn[usCol], trn[mvCol]], trn[rtCol], batch_size=64, nb_epoch=8,
       validation_data=([val[usCol], val[mvCol]], val[rtCol]))

