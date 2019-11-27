# -*- coding: utf-8 -*-
# @Time    : 2019/11/24 15:19
# @Author  : zxl
# @FileName: Linear.py

import tensorflow as tf
import numpy as np

"""
尝试两部分都用线性模型，都当成回归任务，最后再用threshold进行分类
"""

class LinearMTL():
    def __init__(self,iter=100000):
        self.iter=iter

    def fit(self,X,Y):
        """
        :param X: 特征
        :param Y: 多个y
        :return:
        """

        self.X=X.astype(np.float32)
        self.y1=Y[0]#gpa
        self.y2=Y[1]#failed
        self.train()

    def train(self):
        X_d=len(self.X[0])

        b1 = tf.Variable(tf.zeros([1]))
        W1 = tf.Variable(tf.zeros([1,X_d]))
        y1 = tf.matmul(W1,np.transpose(self.X)) + b1

        b2 = tf.Variable(tf.zeros([1]))
        W2 = tf.Variable(tf.zeros([1,X_d]))
        y2 = tf.matmul(W2,np.transpose(self.X)) + b2

        # 方差
        #正则项

        l=0.5
        loss1 = tf.reduce_mean(tf.square(y1 - self.y1.reshape(1,len(self.y1))) + l*tf.nn.l2_loss(W1))
        loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y2.reshape(1,len(self.y2)), logits=y2))

        loss = loss1+2*loss2
        # 构建优化器
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train = optimizer.minimize(loss)

        # 初始化全局变量
        init = tf.global_variables_initializer()

        # 启动图
        with tf.Session() as sess:
            sess.run(init)
            for step in range(1, self.iter):
                print(step)
                sess.run(train)
                sess.run(W1)
                sess.run(b1)
                sess.run(W2)
                sess.run(b2)
                print("loss:%f"%sess.run(loss))
            self.W1=sess.run(W1)
            self.b1=sess.run(b1)
            self.W2=sess.run(W2)
            self.b2=sess.run(b2)


    def predict(self,X):
        """
        :param X:特征
        :return: 多任务上的值
        """
        y1=np.dot(self.W1,np.transpose(X))+self.b1
        y2=np.dot(self.W2,np.transpose(X))+self.b2
        y1=y1.flatten()
        y2=y2.flatten()
        thres=0.5
        y2[y2>thres]=1
        y2[y2<=thres]=0
        res=np.array([y1,y2])
        return res
