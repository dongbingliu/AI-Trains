#!/anaconda3/envs/tensorflow/bin/python  
# -*- coding:utf-8 _*-  
""" 
@author:Jerry Liu
@file: ImageDataGenerator.py 
@version: v1.00.00
@time: 2019/01/05 
@email:liudongbing8@163.com
@function： 数据增强
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def dataPreProcess(dirPath,filePath):

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    image = load_img(filePath)
    x = img_to_array(image)
    x = x.reshape((1,) + x.shape)
    import AiTrainDatas
    i = 0
    if dirPath == "../AiTrainDatas/foods/train/西红柿炒鸡蛋":
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='../AiTrainDatas/newFoods/train/西红柿炒鸡蛋', save_prefix='tomato', save_format='jpeg'):
            i += 1
            if i > 5:
                break


    else:
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='../AiTrainDatas/newFoods/train/鱼', save_prefix='fish', save_format='jpeg'):
            i += 1
            if i > 5:
                break
