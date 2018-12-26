# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:08:36 2018

@author: XieTianwen
"""

import numpy as np
#import copy
#import pickle
#import time

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding
from train_poetry import TextConverter



text = open('datas/poetry.txt', encoding = 'utf-8').read()
converter = TextConverter(text, 3500, 'T.pkl')
vocab_size = converter.vocab_size



def build_samplemodel(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, batch_input_shape = (1, 1)))
    for i in range(2):
        model.add(LSTM(128, return_sequences=(i != 1), stateful = True))
        model.add(Dropout(0.5))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    return model


def pick_top_n(preds, vocab_size, top_n=3):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c




def sample( n_samples, prime, vocab_size):
        samples = [c for c in prime]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            preds = model.predict(x)

        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            
            preds = model.predict(x)
            c = pick_top_n(preds, vocab_size)
            samples.append(c)
        return np.array(samples)

model = build_samplemodel(vocab_size)
model.load_weights('poetry_new.h5')
prime = converter.text_to_arr('月')
arr = sample(300, prime, vocab_size)
print(converter.arr_to_text(arr))

