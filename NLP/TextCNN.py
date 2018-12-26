#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:Jerry Liu
@file: TextCNN.py 
@version:
@time: 2018/12/22 
@email:liudongbing8@163.com
@function： 
"""

import re
import numpy as np
from keras.layers import Conv1D, MaxPool1D, Embedding, Dense, Flatten, Dropout, Concatenate, Input

from keras.models import Model

dim = 100
filt_size = [1, 2, 4]
file_num = 50


# "<PAD/>填充长度空白，可以是其他的字符串，区别其他"
def pad_sentences(sentences, padding_word="<PAD/>"):
    print(type(sentences[0]))
    sequence_length = max((len(x) for x in sentences))
    pad_sentences = []
    for i in range(len(sentences)):
        num_pad = sequence_length - len(sentences[i])
        print(sentences[i])
        new_sentence = sentences[i] + [padding_word] * num_pad
        pad_sentences.append(new_sentence)
    return pad_sentences, sequence_length


def load_data(pos_file, neg_file):
    pos_list = list(open(pos_file).readlines())
    neg_list = list(open(neg_file).readlines())

    all_list = pos_list + neg_list
    print(all_list[:3])
    all_list = [re.findall("[a-zA-Z]+", ele) for ele in all_list]
    print(all_list[:3])

    pos_label = np.asarray([[0, 1] for _ in range(len(pos_list))])
    neg_label = np.asarray([[1, 0] for _ in range(len(neg_list))])

    label = np.concatenate([pos_label, neg_label], axis=0)

    return all_list, label


# load_data(pos_file="datas/pos.txt",neg_file="datas/neg.txt")
#
# print(re.findall("[0-9]+","shgs 88sdfhs989asdfhg908adsfh"))

sentences, labels = load_data(pos_file="datas/pos.txt", neg_file="datas/neg.txt")
pad_sentences, seq_len = pad_sentences(sentences)


# 词频函数
def build_covab(sentences):
    d = {}
    for sentence in sentences:
        for word in sentence:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
    s = sorted(d.items(), key=lambda x: x[1], reverse=True)
    s = [x[0] for x in s]
    word2int = {v: k for k, v in enumerate(s)}
    int2word = {k: v for k, v in enumerate(s)}
    return word2int, int2word


word2int, int2word = build_covab(pad_sentences)
vocab_size = len(word2int)
embedding = np.random.randn(vocab_size, dim)
embedding[0] = np.zeros((1, dim))

x = np.array([[word2int[word] for word in sentence] for sentence in pad_sentences])
y = np.array(labels)

index = np.random.permutation(np.arange(len(x)))
x = x[index]
y = y[index]
train_len = int(len(x) * 0.9)
x_train = x[:train_len]
y_train = y[:train_len]
x_test = x[train_len:]
y_test = y[train_len:]

input_shape = (seq_len,)

model_input = Input(shape=input_shape)
z = Embedding(vocab_size, dim, input_length=seq_len, weights=[embedding])(model_input)
conv_block = []
for sz in filt_size:
    conv = Conv1D(filters=file_num,
                  kernel_size=sz,
                  padding='same',
                  activation='relu')(z)
    conv = MaxPool1D(pool_size=seq_len)(conv)
    conv = Flatten()(conv)
    conv_block.append(conv)
z = Concatenate()(conv_block) if len(conv_block) > 1 else conv_block[0]
z = Dropout(0.5)(z)
z = Dense(50, activation='relu')(z)
model_output = Dense(2, activation='softmax')(z)

model = Model(model_input, model_output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=80, epochs=20,
          validation_data=(x_test, y_test), verbose=1)
