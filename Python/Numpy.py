#!/anaconda3/envs/tensorflow/bin/python  
# -*- coding:utf-8 _*-  
""" 
@author:Jerry Liu
@file: Numpy.py 
@version: v1.00.00
@time: 2018/12/30 
@email:liudongbing8@163.com
@function： 
"""
import math
import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.concatenate((a,b),axis=0))
print(np.column_stack((a,b)))

# print(a.reshape(10,10))
print(list(a))

c=np.array([[1,2],[3,4]])
print(c.tolist())

a = np.asarray(range(100))
print(a[np.where((a%2==0) & (a>50))])

# 产生随机数
# 产生 0-1 随机数
print(np.random.random())
# 产生1-100 之间10个整数
print(np.random.randint(1,100,10))
# 产生随机正太分布函数
print(np.random.shuffle(np.random.randint(1,100,10) ))

#