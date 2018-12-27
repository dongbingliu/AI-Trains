#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:Jerry Liu
@file: MachineLearning.py 
@version: v1.00
@time: 2018/12/11 
@email:liudongbing8@163.com
@function： 获取训练集与测试集数据工具类
"""

import os
import numpy as np
from PIL import Image

img_rows = 50
img_cols = 50


def getClass(dirPath):
    classDirPathList = []
    dirNames = os.listdir(dirPath)
    print(dirNames)

    for dir in dirNames:
        classDirPath = os.path.join(dirPath, dir)
        classDirPathList.append(classDirPath)

    return classDirPathList


def load_data_rgb(dirPath, img_rows, img_cols):
    label = 0
    classDirPathList = getClass(dirPath)
    # print(classDirPathList)
    labelList = []
    imageList = []
    for classDir in classDirPathList:
        if os.path.isdir(classDir):
            imageAllFile = os.listdir(classDir)
            for subFile in imageAllFile:
                img = Image.open(os.path.join(classDir, subFile))
                img = img.resize((img_rows, img_cols))
                img_ndarry = np.asarray(img, dtype="float64") / 256
                if (img_ndarry.shape == (img_rows, img_cols, 3)):
                    imageList.append(img_ndarry)
                    if subFile.find("cat") == 0 :
                        label = 0
                    elif subFile.find("dog") == 0:
                        label = 1

                    labelList.append(label)
            label = label + 1
    featureArray = np.array(imageList)
    labelArray = np.array(labelList)

    return featureArray, labelArray


# 获取菜品所有数据集
def getFoodDatas():
    x_train, y_train = load_data_rgb('../AiTrainDatas/foods/train', img_rows, img_cols)
    x_test, y_test = load_data_rgb('../AiTrainDatas/foods/test', img_rows, img_cols)
    x_train = x_train.reshape(-1, 50 * 50 * 3)
    x_test = x_test.reshape(-1, 50 * 50 * 3)
    return x_train, y_train, x_test, y_test


# 获取猫狗所有数据集
def getAnimalsDatas():
    from sklearn.model_selection import train_test_split
    x_datas, y_datas = load_data_rgb("../AiTrainDatas/animals", img_rows, img_cols)
    x_train, x_test, y_train, y_test = train_test_split(x_datas, y_datas, test_size=0.25, random_state=None)
    x_train = x_train.reshape(-1, 50 * 50 * 3)
    x_test = x_test.reshape(-1, 50 * 50 * 3)
    return x_train, y_train, x_test, y_test


# 获取花卉所有数据集
def getFlowerDatas():
    from sklearn.model_selection import train_test_split
    x_datas, y_datas = load_data_rgb("../AiTrainDatas/flower_photos", img_rows, img_cols)
    x_train, x_test, y_train, y_test = train_test_split(x_datas, y_datas,
                                                        test_size=0.2,
                                                        random_state=None)
    x_train = x_train.reshape(-1, 50 * 50 * 3)
    x_test = x_test.reshape(-1, 50 * 50 * 3)
    return x_train, y_train, x_test, y_test


# 获取大规模物体所有数据集
def getMulObjectDatas():
    from sklearn.model_selection import train_test_split
    x_datas, y_datas = load_data_rgb("../AiTrainDatas/MulObjects", img_rows, img_cols)
    x_train, x_test, y_train, y_test = train_test_split(x_datas, y_datas, test_size=0.25, random_state=None)
    x_train = x_train.reshape(-1, 50 * 50 * 3)
    x_test = x_test.reshape(-1, 50 * 50 * 3)
    return x_train, y_train, x_test, y_test

