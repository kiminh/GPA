# -*- coding: utf-8 -*-
# @Time    : 2019/11/26 21:22
# @Author  : zxl
# @FileName: example-batch.py

"""
batch方式训练
"""

import tensorflow as tf
import scipy.io as sio
import numpy as np


def get_Batch(data, label, batch_size):
    print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32 )
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
    return x_batch, y_batch


# data = sio.loadmat('data.mat')
# train_x = data['train_x']
# train_y = data['train_y']
# test_x = data['test_x']
# test_y = data['test_y']

train_x=np.full(shape=(500,10),fill_value=2)
train_y=np.full(shape=(500,2),fill_value=3)
test_x=np.full(shape=(200,10),fill_value=5)
test_y=np.full(shape=(200,2),fill_value=4)



x = tf.placeholder(tf.float32, [None, 10])
y = tf.placeholder(tf.float32, [None, 2])

w = tf.Variable(tf.truncated_normal([10, 2], stddev=0.1))
b = tf.Variable(tf.truncated_normal([2], stddev=0.1))
pred = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(2e-5).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='evaluation')

x_batch, y_batch = get_Batch(train_x, train_y, 10)
# 训练
with tf.Session() as sess:
    #初始化参数
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 开启协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)
    epoch = 0
    try:
        while not coord.should_stop():
            # 获取训练用的每一个batch中batch_size个样本和标签
            data, label = sess.run([x_batch, y_batch])
            data=np.full(shape=(10,10),fill_value=1)
            label=np.full(shape=(10,2),fill_value=1)
            sess.run(optimizer, feed_dict={x: data, y: label})
            train_accuracy = accuracy.eval({x: data, y: label})
            test_accuracy = accuracy.eval({x: test_x, y: test_y})
            print("Epoch %d, Training accuracy %g, Testing accuracy %g" % (epoch, train_accuracy, test_accuracy))
            epoch = epoch + 1
    except tf.errors.OutOfRangeError:  # num_epochs 次数用完会抛出此异常
        print("---Train end---")
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('---Programm end---')
    coord.join(threads)  # 把开启的线程加入主线程，等待threads结束