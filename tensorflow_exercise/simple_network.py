# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 模型超参数
batch_size = 32
epoch_num = 20000
learning_rate = 0.001
display_num = 500
dropout_rate = 0.5

# 加载数据
mnist = input_data.read_data_sets(r"E:\data", one_hot=True)

# 初始化输入输出占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_tensor')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='output_tensor')

# 定义模型
# 两层全连接，再加上一个输出层
dense_1 = tf.layers.dense(inputs=x, units=256,
                          kernel_initializer=tf.random_normal_initializer(),
                          activation=tf.nn.sigmoid, name='dense_1')
dense_2 = tf.layers.dense(inputs=dense_1, units=256,
                          kernel_initializer=tf.random_normal_initializer(),
                          activation=tf.nn.sigmoid, name='dense_2')
dense_3 = tf.layers.dense(inputs=dense_2, units=10,
                          kernel_initializer=tf.random_normal_initializer(),
                          activation=tf.nn.sigmoid, name='output')

output = tf.nn.softmax(dense_3)

# 定义损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 定义准确率
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 定义权重变量初始化操作
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)

    # for epoch_i in range(epoch_num):
    #     num_steps = int(mnist.train.num_examples / batch_size)

    for steps in range(epoch_num):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if steps % display_num == 0:
            batch_x, batch_y = mnist.test.next_batch(300)
            my_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
            print('Epoch: %d, avg_loss: %f, Accuracy: %f' % (steps, my_loss, batch_accuracy))

    print("Optimization Finished!")

    # 测试模型再测试集上的准确率
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
