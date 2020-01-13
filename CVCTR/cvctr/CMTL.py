# -*- coding: utf-8 -*-
# @Time    : 2020/1/12 15:32
# @Author  : zxl
# @FileName: CMTL.py

import numpy as np
import tensorflow as tf


"""
一个任务预测各个区间内failed概率，另一个任务预测gpa区间
"""


class CMTL:
    def __init__(self):
        pass

    def loss(self,x1,x2):
        """
        评价两个向量差异
        :param x1:
        :param x2:
        :return:
        """


    def fit(self,X,Y,h1=32,h2=32,epoch=5000,learning_rate=0.001,lamda=0.001,batch_size=200,alpha=0.5):
        """
        :param X: 特征， ndarray, n*d
        :param Y: label, ndarray, [train_gpa_y,train_failed_y], train_gpa_y: n*4, train_failed_y: n*2 （挂科，通过）
        :return: None
        """
        self.h_f=h1
        self.h_g=h2
        self.epoch=epoch
        self.learning_rate = learning_rate
        self.lamda = lamda  # 正则项参数
        self.batch_size = batch_size
        self.alpha = alpha  # gpa的权重
        self.train(X,Y)

    def train(self,X,Y):
        d=len(X[0])
        record_num=len(X)
        true_gpa=Y[0]
        true_failed=Y[1]

        placeholder_X = tf.placeholder(tf.float64, [None, d], name="placeholder_X")
        placeholder_gpa = tf.placeholder(tf.float64, [None, 4], name="placeholder_gpa")
        placeholder_failed = tf.placeholder(tf.float64, [None, 2], name="placeholder_failed")

        #左边failed网络
        Wf1 = tf.Variable(tf.ones([d, self.h_f],dtype=tf.float64))
        Wf2 = tf.Variable(tf.ones([self.h_f,4],dtype=tf.float64))
        bf1=tf.Variable(tf.ones([1],dtype=tf.float64))
        bf2=tf.Variable(tf.ones([1],dtype=tf.float64))
        Hf1 = tf.matmul(placeholder_X,Wf1)+bf1
        Hf1=tf.nn.relu(Hf1)
        Hf2 = tf.matmul(Hf1,Wf2)+bf2
        Hf2 = tf.nn.softmax(Hf2)

        #右边gpa网络
        Wg1=tf.Variable(tf.ones([d,self.h_g],dtype=tf.float64))
        Wg2=tf.Variable(tf.ones([self.h_g,4],dtype=tf.float64))
        bg1=tf.Variable(tf.ones([1],dtype=tf.float64))
        bg2=tf.Variable(tf.ones([1],dtype=tf.float64))
        Hg1=tf.matmul(placeholder_X,Wg1)+bg1
        Hg1=tf.nn.relu(Hg1)
        predict_gpa=tf.matmul(Hg1,Wg2)+bg2
        predict_gpa=tf.nn.softmax(predict_gpa)

        #综合两个结果
        failed_rate=tf.reduce_sum(Hf2*predict_gpa,axis=1,keep_dims=True)
        pass_rate=tf.reduce_sum((1-Hf2)*predict_gpa,axis=1,keep_dims=True)
        predict_failed=tf.concat([failed_rate,pass_rate],axis=1)

        loss1=tf.reduce_mean(tf.square(predict_gpa-placeholder_gpa))
        loss2 = tf.reduce_mean(tf.square(predict_failed - placeholder_failed))
        loss_reg=tf.reduce_mean(tf.nn.l2_loss(Wf1)+tf.nn.l2_loss(Wf2)+tf.nn.l2_loss(Wg1)+tf.nn.l2_loss(Wg2))
        loss=self.alpha*loss1+(1-self.alpha)*loss2+self.lamda*loss_reg

        # 构建优化器
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        batch_size = self.batch_size
        # 启动图
        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for e in np.arange(0, self.epoch):
                cur_loss = 0.0
                i = 0
                while i < record_num:

                    cur_X = X[i:min(i + self.batch_size, record_num), :]
                    cur_gpa = true_gpa[i:min(i + batch_size, record_num), :]
                    cur_failed = true_failed[i:min(i + batch_size, record_num), :]

                    sess.run(train, feed_dict={placeholder_X: cur_X, placeholder_gpa: cur_gpa, placeholder_failed: cur_failed})
                    cur_loss += sess.run(loss, feed_dict={placeholder_X: cur_X, placeholder_gpa: cur_gpa, placeholder_failed: cur_failed})
                    i = min(i + self.batch_size, record_num)

                print("epoch:%d, loss:%f" % (e, cur_loss))

            self.Wf1=sess.run(Wf1)
            self.Wf2=sess.run(Wf2)
            self.bf1=sess.run(bf1)
            self.bf2=sess.run(bf2)
            self.Wg1=sess.run(Wg1)
            self.Wg2=sess.run(Wg2)
            self.bg1=sess.run(bg1)
            self.bg2=sess.run(bg2)
            print("epoch:%d,batch_size:%d,failed_hidden:%d,gpa_hidden:%d,learning_rate:%f,lambda:%f, alpha:%f" % (
            self.epoch, self.batch_size, self.h_f,self.h_g, self.learning_rate, self.lamda, self.alpha))

    def predict(self,X):
        sess = tf.Session()
        with sess.as_default():
            Hf1=tf.matmul(X,self.Wf1)+self.bf1
            Hf1=tf.nn.relu(Hf1)
            Hf2=tf.matmul(Hf1,self.Wf2)+self.bf2
            Hf2=tf.nn.softmax(Hf2)

            Hg1=tf.matmul(X,self.Wg1)+self.bg1
            Hg1=tf.nn.relu(Hg1)
            predict_gpa=tf.matmul(Hg1,self.Wg2)+self.bg2
            predict_gpa=tf.nn.softmax(predict_gpa)

            failed_rate = tf.reduce_sum(Hf2 * predict_gpa,axis=1,keep_dims=True)
            pass_rate = tf.reduce_sum((1 - Hf2) * predict_gpa,axis=1,keep_dims=True)
            predict_failed = tf.concat([failed_rate, pass_rate],axis=1)

            predict_gpa=predict_gpa.eval()
            predict_failed=predict_failed.eval()

        res=[predict_gpa,predict_failed]
        return res


