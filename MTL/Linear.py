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
    def __init__(self,epoch=1):
        self.h = 50
        self.epoch = epoch
        self.learning_rate = 0.001
        self.lamda = 0.0001  # 正则项参数
        self.batch_size = 200
        self.alpha = 0.7  # gpa的权重


    def fit(self,X,Y):
        """
        :param X: 特征
        :param Y: 多个y
        :return:
        """

        self.X=X.astype(np.float32)
        self.y1=np.reshape(Y[0],newshape=(len(Y[0]),1))#gpa
        self.y2=np.reshape(Y[1],newshape=(len(Y[1]),1))#failed
        self.train()

    def train(self):
        X_d=len(self.X[0])
        record_num=len(self.X)

        X = tf.placeholder(tf.float32, [None, X_d])
        y_gpa = tf.placeholder(tf.float32, [None,1])
        y_failed = tf.placeholder(tf.float32, [None,1])

        b1 = tf.Variable(tf.zeros([1]))
        W1 = tf.Variable(tf.zeros([1,X_d]))
        y1 = tf.matmul(X,tf.transpose(W1)) + b1

        b2 = tf.Variable(tf.zeros([1]))
        W2 = tf.Variable(tf.zeros([1,X_d]))
        y2 = tf.matmul(X,tf.transpose(W2)) + b2
        #是否通过增加relu激活函数
        y2=tf.nn.relu(y2)

        # 方差
        #正则项
        reg_loss1=tf.reduce_mean(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2))
        reg_loss2=tf.reduce_mean(tf.nn.l2_loss(b1)+tf.nn.l2_loss(b2))

        loss1 = tf.reduce_mean(tf.square(y1 - y_gpa) )
        loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_failed, logits=y2))

        loss = self.alpha*loss1+(1-self.alpha)*loss2+self.lamda*(reg_loss1+reg_loss2)
        # 构建优化器
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        # 初始化全局变量
        init = tf.global_variables_initializer()

        batch_size=self.batch_size
        # 启动图
        with tf.Session() as sess:
            sess.run(init)
            for step in range(1, self.epoch):
                i = 0
                cur_loss=0.0
                while i < record_num:
                    cur_X = self.X[i:min(i + batch_size, record_num),:]
                    cur_gpa = self.y1[i:min(i + batch_size, record_num),:]
                    cur_failed = self.y2[i:min(i + batch_size, record_num),:]

                    sess.run(train, feed_dict={X: cur_X, y_gpa: cur_gpa, y_failed: cur_failed})
                    cur_loss += sess.run(loss, feed_dict={X: cur_X, y_gpa: cur_gpa, y_failed: cur_failed})

                    i = min(i + batch_size, record_num)
                print("epoch:%d, loss:%f"%(step,cur_loss))
            self.W1=sess.run(W1)
            self.b1=sess.run(b1)
            self.W2=sess.run(W2)
            self.b2=sess.run(b2)
            print("hidden:%d,learning_rate:%f,lambda:%f, alpha:%f" % (self.h, self.learning_rate, self.lamda, self.alpha))


    def predict(self,X):
        """
        :param X:特征
        :return: 多任务上的值
        """
        y1=np.dot(X,np.transpose(self.W1))+self.b1
        y2=np.dot(X,np.transpose(self.W2))+self.b2
        y1=y1.flatten()
        y2=y2.flatten()
        thres=0.5
        y2[y2>thres]=1
        y2[y2<=thres]=0
        res=np.array([y1,y2])
        return res
