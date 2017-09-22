# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
rng = np.random

# 设置参数
learning_rate = 0.01
training_epochs = 100
batch_size = 32
display_step = 1

# 导入 MNIST 数据
mnist = input_data.read_data_sets(r"E:\data", one_hot=True)

# 定义输入输出占位符
train_X = tf.placeholder(tf.float32, shape=[None, 784])
train_Y = tf.placeholder(tf.float32, shape=[None, 10])

# 定义权重向量
W = tf.Variable(tf.random_normal(shape=[784, 10]), name='weights')
B = tf.Variable(tf.random_normal(shape=[10]), name='bias')

predict_y = tf.nn.softmax(tf.add(tf.matmul(train_X, W), B))

# 定义模型损失函数，采用交叉熵:-y*log(y`)
loss = tf.reduce_mean(-tf.reduce_sum(train_Y*tf.log(predict_y), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_loss = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch_i in range(total_batch):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, c = sess.run([optimizer, loss], feed_dict={train_X: batch_x, train_Y: batch_y})

            avg_loss += c / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), ",loss=", "{:.9f}".format(avg_loss))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(train_Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({train_X: mnist.test.images, train_Y: mnist.test.labels}))