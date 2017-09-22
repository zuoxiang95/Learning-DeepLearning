# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


plt.ion()
# 生成数据
n_observations = 100
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
ys = 5 * xs + 4 + np.random.uniform(-0.3, 0.3, n_observations)
ax.plot(xs, ys)

plt.draw()
fig.show()

# # 设置占位符
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# # 创建权重和偏移量
# w = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
# # 设置计算输出方法
# Y_predict = tf.add(tf.multiply(X, w), b)
#
# # 定义损失函数
# loss = tf.reduce_sum(tf.pow(Y_predict - ys, 2)) / (n_observations - 1)
#
# # 学习率
# learning_rate = 0.01
# # 设置优化方法：使用梯度下降优化损失
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss)
#
# # 设置迭代次数
# n_epochs = 1000
# with tf.Session() as sess:
#     # 初始化所有变量
#     sess.run(tf.global_variables_initializer())
#
#     pre_training_loss = 0
#     for epoch_i in range(n_epochs):
#         for (x, y) in zip(xs, ys):
#             sess.run(optimizer, feed_dict={X: x, Y: y})
#         training_loss = sess.run(loss, feed_dict={X: xs, Y: ys})
#         print(training_loss)
#         if epoch_i % 20 == 0:
#             ax.plot(xs, Y_predict.eval(feed_dict={X: xs}), 'k', alpha=epoch_i / n_epochs)
#         #     fig.show()
#         #     plt.draw()
#
#         if np.abs(pre_training_loss - training_loss) < 0.0001:
#             break
#         pre_training_loss = training_loss
#     print(sess.run(w))
#     print(sess.run(b))
# # fig.show()
# # plt.waitforbuttonpress()
