# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:13:17 2018

@author: XieTianwen
"""

import numpy as np
import copy
import pickle
import time
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
# from keras.utils import to_categorical
from keras.optimizers import Adam, SGD

n_seqs = 32
n_steps = 26
max_steps = 1000


class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            #vocabulary 词汇，set()集合 python 内置函数，去重；
            vocab = set(text)
            self.vacab_tmp = vocab
            # max_vocab_process，vocab_count 字典
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                # 字的重复次数，字典索引值即字的重复次数
                vocab_count[word] += 1
            self.vocab_count_00 = vocab_count
            vocab_count_list = []
            # 字典转换为 list，list 元素为元组
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            self.vocab_count_list_tmp01 = vocab_count_list
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            self.vocab_count_list_tmp02 = vocab_count_list
            if len(vocab_count_list) > max_vocab:
                # 砍掉 max 后的偏僻字
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))
        self.vocab_size = len(self.vocab) + 1

    def word_to_int(self, word):
        # 判断字 word 是否在字典word_to_int_table中，返回索引值
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        #        else:
        #            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)


def batch_generator(arr, n_seqs, n_steps, vocab_size):
    sess = tf.Session()
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            y = tf.one_hot(y, vocab_size).eval(session=sess)
            yield x, y


def build_model(n_seqs, n_steps, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, batch_input_shape=(n_seqs, n_steps)))
    for i in range(2):
        model.add(LSTM(128, return_sequences=True, stateful=True))
        model.add(Dropout(0.5))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    model.layers[0].trainable = False
    return model


def main():
    text = open('datas/poetry.txt', encoding='utf-8').read()
    converter = TextConverter(text, 3500)
    converter.save_to_file('T.pkl')
    arr = converter.text_to_arr(text)
    vocab_size = converter.vocab_size

    model = build_model(n_seqs=n_seqs, n_steps=n_steps, vocab_size=vocab_size)
    optim = Adam(clipnorm=5)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    g = batch_generator(arr, n_seqs=n_seqs, n_steps=n_steps, vocab_size=vocab_size)

    step = 0

    for x, y in g:
        # print("x=",x)
        # print("y=",y)
        start = time.time()
        loss, acc = model.train_on_batch(x, y)
        step = step + 1
        end = time.time()

        if step % 10 == 0:
            print('step: {}, acc: {}, loss: {}, sec/batch:{}'.format(step, acc, loss, (end - start)))
            model.save_weights('poetry_new.h5')
        if step == max_steps:
            break


if __name__ == '__main__':
    main()
