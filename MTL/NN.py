# -*- coding: utf-8 -*-
# @Time    : 2019/11/25 20:34
# @Author  : zxl
# @FileName: NN.py

import numpy as np
import tensorflow as tf

class nnMTL():
    """
    神经网络多任务
    先共享隐层，再各自预测
    """
    def __init__(self,h=50,iter=5000):
        self.h=h
        self.iter=iter

    def fit(self,X,Y):
        """

        :param X: 特征向量
        :param Y: Y[0]:gpa，Y[1]:failed
        :return:
        """
        self.X=X
        self.y1=Y[0]
        self.y2=Y[1]
        self.train()

    def train(self):
        d=len(self.X[0])

        W=tf.Variable(tf.ones([d,self.h]))
        H=tf.matmul(self.X,W)
        #TODO 隐藏层激活函数未写，不过隐层应该也不需要激活函数吧
        H=tf.nn.relu(H)

        b1 = tf.Variable(tf.zeros([1]))
        b2 = tf.Variable(tf.zeros([1]))

        W1 = tf.Variable(tf.zeros([1, self.h]))
        W2 = tf.Variable(tf.zeros([1, self.h]))

        y1 = tf.matmul(W1, tf.transpose(H)) + b1
        y2 = tf.matmul(W2, tf.transpose(H)) + b2
        #TODO 输出层激活函数未写
        # y1=tf.nn.relu(y1)
        # y2=tf.nn.sigmoid(y2)

        # 方差
        # 正则项

        l = 0.5
        reg_loss=tf.reduce_mean(0.001*(tf.nn.l2_loss(W)+tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)))
        loss1 = tf.reduce_mean(tf.square(y1 - self.y1.reshape(1, len(self.y1))) )
        loss2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y2.reshape(1, len(self.y2)), logits=y2))

        loss = 2*loss1 +  loss2+reg_loss

        # 构建优化器
        optimizer=tf.train.AdamOptimizer(0.05)
        # optimizer = tf.train.GradientDescentOptimizer(0.001)
        train = optimizer.minimize(loss)

        # 初始化全局变量
        init = tf.global_variables_initializer()

        # 启动图
        with tf.Session() as sess:
            sess.run(init)
            for step in range(0, self.iter):
                print(step)
                sess.run(train)
                sess.run(W)
                sess.run(W1)
                sess.run(b1)
                sess.run(W2)
                sess.run(b2)
                print("loss:%f" % sess.run(loss))

            self.W=sess.run(W)
            self.W1 = sess.run(W1)
            self.b1 = sess.run(b1)
            self.W2 = sess.run(W2)
            self.b2 = sess.run(b2)


    def predict(self,X):
        """
        :param X:特征
        :return: 多任务上的值
        """
        H=np.dot(X,self.W)
        y1 = np.dot(self.W1, np.transpose(H)) + self.b1
        y2 = np.dot(self.W2, np.transpose(H)) + self.b2
        y1 = y1.flatten()
        y2 = y2.flatten()
        thres = 0.5
        y2[y2 > thres] = 1
        y2[y2 <= thres] = 0
        res = np.array([y1, y2])
        return res
