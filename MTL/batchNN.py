# -*- coding: utf-8 -*-
# @Time    : 2019/11/26 20:52
# @Author  : zxl
# @FileName: batchNN.py

import numpy as np
import tensorflow as tf

"""
NN的batch实现版本
"""

class nnMTL2():
    """
    神经网络多任务
    先共享隐层，再各自预测
    """
    def __init__(self,h=50,epoch=100):
        self.h=h
        self.epoch=epoch
        self.learning_rate=0.001
        self.lamda=0.001#正则项参数
        self.batch_size=200
        self.alpha=0.7#gpa的权重

    def fit(self,X,Y):
        """

        :param X: 特征向量
        :param Y: Y[0]:gpa，Y[1]:failed
        :return:
        """
        self.X=X
        self.Y=np.transpose(Y)#N*2
        self.y1=np.reshape(Y[0],newshape=(len(Y[0]),1))#gpa
        self.y2=np.reshape(Y[1],newshape=(len(Y[1]),1))#failed
        self.train()


    def train(self):
        d=len(self.X[0])
        record_num=len(self.X)

        X=tf.placeholder(tf.float32,[None,d],name="X")
        y_gpa=tf.placeholder(tf.float32,[None,1],name="y_gpa")
        y_failed=tf.placeholder(tf.float32,[None,1],name="y_failed")

        W=tf.Variable(tf.ones([d,self.h]))
        H=tf.matmul(X,W)
        #TODO 隐藏层激活函数未写，不过隐层应该也不需要激活函数吧
        H=tf.nn.relu(H)

        b1 = tf.Variable(tf.zeros([1]))
        b2 = tf.Variable(tf.zeros([1]))
        W1 = tf.Variable(tf.zeros([1, self.h]))
        W2 = tf.Variable(tf.zeros([1, self.h]))

        y1 = tf.matmul(H,tf.transpose(W1)) + b1
        y2 = tf.matmul(H, tf.transpose(W2)) + b2
        #TODO 输出层激活函数未写
        # y1=tf.nn.dropout(y1,keep_prob=1)
        # y2=tf.nn.sigmoid(y2)

        # 方差
        # 正则项
        reg_loss1=tf.reduce_mean((tf.nn.l2_loss(W)+tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)))
        reg_loss2=tf.reduce_mean(tf.nn.l2_loss(b1)+tf.nn.l2_loss(b2))

        loss1 = tf.reduce_mean(tf.square(y1 - y_gpa) )

        #loss2试试sigmoid+mse
        loss2=tf.reduce_mean(tf.square(tf.nn.sigmoid(y2)-y_failed))
        # loss2 = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_        with_logits(labels=y_failed, logits=y2))

        loss = self.alpha*loss1 + (1.0-self.alpha)*loss2+ self.lamda*(reg_loss1+reg_loss2)

        # 构建优化器
        optimizer=tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)


        batch_size=self.batch_size
        # 启动图
        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for e in np.arange(0,self.epoch):
                cur_loss=0.0
                i=0
                while i<record_num:
                    cur_X=self.X[i:min(i+self.batch_size,record_num),:]
                    cur_gpa=self.y1[i:min(i+batch_size,record_num),:]
                    cur_failed=self.y2[i:min(i+batch_size,record_num),:]

                    sess.run(train,feed_dict={X:cur_X,y_gpa:cur_gpa,y_failed:cur_failed})
                    cur_loss+=sess.run(loss,feed_dict={X:cur_X,y_gpa:cur_gpa,y_failed:cur_failed})

                    i=min(i+self.batch_size,record_num)
                print("epoch:%d, loss:%f"%(e,cur_loss))

            # #保存模型
            # saver=tf.train.Saver()
            # self.save_path=saver.save(sess,'models/my_model',global_step=1000)

            self.W=sess.run(W)
            self.W1 = sess.run(W1)
            self.b1 = sess.run(b1)
            self.W2 = sess.run(W2)
            self.b2 = sess.run(b2)
        print("epoch:%d,batch_size:%d,hidden:%d,learning_rate:%f,lambda:%f, alpha:%f" % (self.epoch,self.batch_size,self.h, self.learning_rate, self.lamda,self.alpha))
        # print("无隐层")

    def predict(self,X):
        """
        :param X:特征
        :return: 多任务上的值
        """
        H=np.dot(X,self.W)
        sess=tf.Session()
        with sess.as_default():
            H=tf.nn.relu(H).eval()

        y1 = np.dot(self.W1, np.transpose(H)) + self.b1
        y2 = np.dot(self.W2, np.transpose(H)) + self.b2
        y1 = y1.flatten()
        y2 = y2.flatten()
        thres = 0.5
        y2[y2 >= thres] = 1
        y2[y2 < thres] = 0
        res = np.array([y1, y2])
        # sess=tf.Session()
        # saver=tf.train.import_meta_graph("models/my_model-1000.meta")
        # saver.restore(sess,tf.train.latest_checkpoint('models/'))
        # fake_y1=np.full(shape=(len(X),1),fill_value=1)
        # fake_y2=np.full(shape=(len(X),1),fill_value=1)
        # graph=tf.get_default_graph()
        # X_fea=graph.get_operation_by_name("X").outputs[0]
        # gpa=graph.get_operation_by_name("y_gpa").outputs[0]
        # failed=graph.get_operation_by_name("y_failed").outputs[0]
        # res_gpa=sess.run(gpa,feed_dict={X_fea:X,gpa:fake_y1,failed:fake_y2})
        # res_failed=sess.run(failed,feed_dict={X_fea:X,gpa:fake_y1,failed:fake_y2})
        # res_failed[res_failed>=0.5]=1
        # res_failed[res_failed<0.5]=1
        # print(res_gpa.shape)
        # print(res_failed.shape)
        # res=[res_gpa,res_failed]

        return res
