# -*- coding: utf-8 -*-
# @Time    : 2019/11/24 15:08
# @Author  : zxl
# @FileName: mtl-eg-simple.py


import tensorflow as tf
import numpy as np

"""
fake data
"""

x_data = np.float32(np.random.rand(2,100))
y1_data=np.dot([0.1,0.2], x_data)+0.3
y2_data=np.dot([0.5,0.9],x_data)+3

"""
线性模型
"""

b1=tf.Variable(tf.zeros([1]))
W1=tf.Variable(tf.random_uniform([1,2],-1,1.0))
y1=tf.matmul(W1,x_data)+b1

b2=tf.Variable(tf.zeros([1]))
W2=tf.Variable(tf.random_uniform([1,2],-1,1))
y2=tf.matmul(W2,x_data)+b2

#方差
loss1=tf.reduce_mean(tf.square(y1-y1_data))
loss2=tf.reduce_mean(tf.square(y2-y2_data))

loss=loss1+loss2
#构建优化器
optimizer=tf.train.GradientDescentOptimizer(0.05)
train=optimizer.minimize(loss)

#初始化全局变量
init=tf.global_variables_initializer()

#启动图
with tf.Session() as sess:
    sess.run(init)
    for step in range(1,300):
        sess.run(train)
        print(step,'W1,b1,W2,b2: ',sess.run(W1),sess.run(b1),sess.run(W2),sess.run(b2))
        print("loss:%f"%sess.run(loss))
