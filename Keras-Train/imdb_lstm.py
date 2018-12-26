#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:Jerry Liu
@file: imdb_lstm.py 
@version:
@time: 2018/12/25 
@email:liudongbing8@163.com
@function： IMDB 情绪分类任务，使用 LSTM 训练
https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
"""
import keras
from keras import Sequential
from keras.datasets import imdb
from keras.layers import Embedding, LSTM, Dense
from keras_preprocessing import sequence

max_features = 20000

max_len =80
batch_size = 32

print("Loading data .....\n")

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)

print("Pad Sequences ....\n")
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)

print("***Build Model***\n")

model = Sequential()
model.add(Embedding(max_features,128))
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer="adam",
              metrics=["accuracy"])

print("Training ...\n")

model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test,y_test))

score,acc = model.evaluate(x_test,y_test,
                           batch_size=batch_size)

print("Test Score : ",score)
print("Test accuracy : ",acc)


