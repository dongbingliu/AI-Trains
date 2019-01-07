'''''Train a simple convnet on the part olivetti faces dataset. 
 
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py 
 
Get to 95% test accuracy after 25 epochs (there is still a lot of margin for parameter tuning). 
'''

from __future__ import print_function
import numpy

numpy.random.seed(1337)  # for reproducibility  

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Input, BatchNormalization, AveragePooling2D, concatenate, ZeroPadding2D
from keras.layers import add

from keras.optimizers import SGD
from keras.utils import np_utils

import os


# There are 40 different classes  
# input image dimensions  
# number of convolutional filters to use  

def readClass(dirPath):
    classDirPathList = []
    dirName = os.listdir(dirPath)
    for d in dirName:
        classDirPath = os.path.join(dirPath, d)
        classDirPathList.append(classDirPath)
    return classDirPathList

import KerasTrain.ImageDataGenerator as ImageGenerator
import shutil
import AiTrainDatas
def dataPreProcess(dirPath):
    # shutil.copy(os.listdir(dirPath),"../AiTrainDatas/new/1")
    classDirPathList = readClass(dirPath)
    for subDirPath in classDirPathList:
        imageAllFile = os.listdir(subDirPath)
        for fileName in imageAllFile:
            filePathName = os.path.join(subDirPath,fileName)
            ImageGenerator.dataPreProcess(subDirPath,filePathName)



def load_data_RGB(dirPath1, dirPath2, dirPath3, img_rows, img_cols, nb_classes):



    count = 0
    classDirPathList = readClass(dirPath1)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 3)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy1 = numpy.array(imageList)
    labelNumpy1 = numpy.array(labelList)

    count = 0
    classDirPathList = readClass(dirPath2)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 3)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy2 = numpy.array(imageList)
    labelNumpy2 = numpy.array(labelList)

    count = 0
    classDirPathList = readClass(dirPath3)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 3)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy3 = numpy.array(imageList)
    labelNumpy3 = numpy.array(labelList)

    labelNumpy1 = labelNumpy1.astype('int64')
    labelNumpy2 = labelNumpy2.astype('int64')
    labelNumpy3 = labelNumpy3.astype('int64')

    labelNumpy1 = np_utils.to_categorical(labelNumpy1, nb_classes)
    labelNumpy2 = np_utils.to_categorical(labelNumpy2, nb_classes)
    # labelNumpy3 = np_utils.to_categorical(labelNumpy3, nb_classes)

    featureNumpy1 = featureNumpy1.reshape(featureNumpy1.shape[0], img_rows, img_cols, 3)
    featureNumpy2 = featureNumpy2.reshape(featureNumpy2.shape[0], img_rows, img_cols, 3)
    featureNumpy3 = featureNumpy3.reshape(featureNumpy3.shape[0], img_rows, img_cols, 3)

    rval = [(featureNumpy1, labelNumpy1), (featureNumpy2, labelNumpy2), (featureNumpy3, labelNumpy3)]

    return rval


def load_predict_data_RGB(dataPath, img_rows, img_cols, nb_classes):
    imageList = []
    img = Image.open(dataPath)
    img = img.resize((img_rows, img_cols))
    img_ndarray = numpy.asarray(img, dtype='float64') / 256
    if (img_ndarray.shape == (img_rows, img_cols, 3)):
        imageList.append(img_ndarray)
    imageList = numpy.array(imageList)
    imageList = imageList.reshape(imageList.shape[0], img_rows, img_cols, 3)
    return imageList


def load_data_Grey(dirPath1, dirPath2, dirPath3, img_rows, img_cols, nb_classes):
    count = 0
    classDirPathList = readClass(dirPath1)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 1)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy1 = numpy.array(imageList)
    labelNumpy1 = numpy.array(labelList)

    count = 0
    classDirPathList = readClass(dirPath2)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 1)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy2 = numpy.array(imageList)
    labelNumpy2 = numpy.array(labelList)

    count = 0
    classDirPathList = readClass(dirPath3)
    labelList = []
    imageList = []
    for c in classDirPathList:
        imageAllFile = os.listdir(c)
        for i in imageAllFile:
            img = Image.open(os.path.join(c, i))
            img = img.resize((img_rows, img_cols))
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            if (img_ndarray.shape == (img_rows, img_cols, 1)):
                imageList.append(img_ndarray)
                labelList.append(count)
        count = count + 1
    featureNumpy3 = numpy.array(imageList)
    labelNumpy3 = numpy.array(labelList)

    labelNumpy1 = labelNumpy1.astype('int64')
    labelNumpy2 = labelNumpy2.astype('int64')
    labelNumpy3 = labelNumpy3.astype('int64')

    labelNumpy1 = np_utils.to_categorical(labelNumpy1, nb_classes)
    labelNumpy2 = np_utils.to_categorical(labelNumpy2, nb_classes)
    # labelNumpy3 = np_utils.to_categorical(labelNumpy3, nb_classes)

    featureNumpy1 = featureNumpy1.reshape(featureNumpy1.shape[0], img_rows, img_cols, 3)
    featureNumpy2 = featureNumpy2.reshape(featureNumpy2.shape[0], img_rows, img_cols, 3)
    featureNumpy3 = featureNumpy3.reshape(featureNumpy3.shape[0], img_rows, img_cols, 3)

    rval = [(featureNumpy1, labelNumpy1), (featureNumpy2, labelNumpy2), (featureNumpy3, labelNumpy3)]

    return rval


def Image_Classification_model_CNN(lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=50, img_cols=50,
                                   RGB=True):
    if (RGB == True):
        color = 3
    elif (RGB == False):
        color = 1
    model = Sequential()
    model.add(Conv2D(30, 5, 5, input_shape=(img_rows, img_cols, color)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(30, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def Image_Classification_model_LeNet(lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=50, img_cols=50,
                                     RGB=True):
    if (RGB == True):
        color = 3
    elif (RGB == False):
        color = 1
    model = Sequential()
    model.add(
        Conv2D(32, (5, 5), strides=(1, 1), input_shape=(img_rows, img_cols, color), padding='valid', activation='relu',
               kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def Image_Classification_model_VGG_13(lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=50, img_cols=50,
                                      RGB=True):
    if (RGB == True):
        color = 3
    elif (RGB == False):
        color = 1

    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_rows, img_cols, color), padding='same', activation='relu',
               kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def Image_Classification_model_VGG_19(lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=50, img_cols=50,
                                      RGB=True):
    if (RGB == True):
        color = 3
    elif (RGB == False):
        color = 1

    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_rows, img_cols, color), padding='same', activation='relu',
               kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def Image_Classification_model_ResNet_34(lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=50, img_cols=50,
                                         RGB=True):
    if (RGB == True):
        color = 3
    elif (RGB == False):
        color = 1

    inpt = Input(shape=(img_rows, img_cols, color))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # (56,56,64)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    # (28,28,128)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    # (14,14,256)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    # (7,7,512)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def train_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, model_url):
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_val, Y_val),
              shuffle=True)
    model.save_weights(model_url, overwrite=True)
    return model


def test_model(X, Y, model_url):
    model.load_weights(model_url)
    classes = model.predict_classes(X, verbose=0)
    test_accuracy = numpy.mean(numpy.equal(Y, classes))

    print("test accuarcy:", test_accuracy)


def test_model_NonSequential(X, Y, model_url):
    model.load_weights(model_url)

    classes = numpy.argmax(model.predict(X), axis=1)
    test_accuracy = numpy.mean(numpy.equal(Y, classes))
    for i in range(len(classes)):
        if Y[i] != classes[i]:
            print(i)
    print("test accuarcy:", test_accuracy)


def predict_model(model, X, model_url):
    model.load_weights(model_url)
    classes = model.predict_classes(X, verbose=0)
    return classes


def predict_model_NonSequential(model, X, model_url):
    model.load_weights(model_url)
    predictMatrix = model.predict(X)
    # numpy.argmax(y_pred1,axis=1)
    return predictMatrix


if __name__ == '__main__':

    dataPreProcess("../AiTrainDatas/foods/train")


    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data_RGB('../AiTrainDatas/foods/train',
                                                                         '../AiTrainDatas/foods/val',
                                                                         '../AiTrainDatas/foods/test', 50, 50, 2)
    model = Image_Classification_model_ResNet_34(nb_classes=2, img_rows=50, img_cols=50)
    # train_model(model, X_train, y_train, X_val, y_val, 40, 50, 'model_weights.h5')
    test_model_NonSequential(X_test, y_test, 'model_weights.h5')


    pic = load_predict_data_RGB('../AiTrainDatas/foods/2/1.jpg', 50, 50, 2)



    result = predict_model_NonSequential(model, pic, 'model_weights.h5')
    print(result)
