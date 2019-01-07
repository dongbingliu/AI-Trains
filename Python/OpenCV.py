#!/anaconda3/envs/tensorflow/bin/python  
# -*- coding:utf-8 _*-  
""" 
@author:Jerry Liu
@file: OpenCV.py 
@version: v1.00.00
@time: 2018/12/30 
@email:liudongbing8@163.com
@function： OpenCv 代码
"""

import cv2
cv2.namedWindow("aaa",cv2.WINDOW_NORMAL)
color_img = cv2.imread("../AiTrainDatas/animals/train/cat.0.jpg",cv2.IMREAD_GRAYSCALE)
# print(color_img.shape)

cv2.imshow("aaa",color_img)
cv2.imwrite("file.jpg",color_img)

cv2.waitKey(0)