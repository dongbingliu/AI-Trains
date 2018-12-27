#!/anaconda3/envs/tensorflow/bin/python  
# -*- coding:utf-8 _*-  
""" 
@author:Jerry Liu
@file: PlotTrainUtils.py 
@version: v1.00.00
@time: 2018/12/27 
@email:liudongbing8@163.com
@function： Keras model.fit 训练数据可视化 Utils
"""
import matplotlib.pyplot as plt

def training_vis(history):
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    acc = history.history['acc']
    # val_acc = history.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    # ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    # ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()
