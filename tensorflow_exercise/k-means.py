# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib.factorization import KMeans
import pdb

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:\data", one_hot=True)
full_data = mnist.train.images

# 设置模型参数
num_steps = 300
batch_size = 1024
k = 10
num_classes = 10
num_feature = 784

# 定义输入输出占位符
x = tf.placeholder(tf.float32, shape=[None, num_feature], name='input_x')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='output_y')

# 使用kmeans函数定义模型
kmeans = KMeans(inputs=x, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

pdb.set_trace()

# 创建 Kmeans 图
(all_source, cluster_idx, scores, cluster_center_initialized, ini_op, train_op) = kmeans.training_graph()
#
cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(ini_op, feed_dict={x: full_data})

    for i in range(1, num_steps + 1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={x: full_data})

        if i % 10 == 0 or i == 1:
            print("Step: %d, Avg distance: %f" % (i, d))
    counts = np.zeros(shape=[k, num_classes])
    for i in range(len(idx)):
        counts[idx[i]] += mnist.train.labels[i]
    labels_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)

    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)

    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_x, test_y = mnist.test.next_batch(batch_size)
    print("Test accuracy:", sess.run(accuracy_op, feed_dict={x: test_x, y: test_y}))

