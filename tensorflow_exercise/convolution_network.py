# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设置模型参数
batch_size = 128
num_step = 2000
display_step = 500
learning_rate = 0.01

# 加载数据
mnist = input_data.read_data_sets(r"E:\data", one_hot=True)


# 定义模型结构
def conv_net(x_dict, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']
        # 原始数据的形状是 1*784，我们需要将其修改为 28*28*1
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # 第一层卷积
        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], activation=tf.nn.relu)
        # 池化层
        max_pooling_1 = tf.layers.max_pooling2d(conv1, 2, 2)
        # 第二层卷积
        conv2 = tf.layers.conv2d(inputs=max_pooling_1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
        # 池化层
        max_pooling_2 = tf.layers.max_pooling2d(conv2, 2, 2)
        # 将二维输出矩阵flatten为1-D用于全连接层
        fc1_input = tf.contrib.layers.flatten(max_pooling_2)
        # 全连接层
        dense_1 = tf.layers.dense(inputs=fc1_input, units=1024, activation=tf.nn.sigmoid)
        # dropout层
        dense_2 = tf.layers.dropout(dense_1, rate=0.5, training=is_training)
        # output层
        output = tf.layers.dense(dense_2, 10)

    return output


def model_fn(features, labels, mode):
    train_logits = conv_net(x_dict=features, reuse=False, is_training=True)
    test_logits = conv_net(x_dict=features, reuse=True, is_training=False)

    # 预测输出结果
    prediction = tf.argmax(test_logits, axis=1)
    # 预测输出概率
    prediction_porb = tf.nn.softmax(test_logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)

    # 定义损失函数
    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(labels, dtype=tf.int32),
            logits=train_logits
        )
    )

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=tf.train.get_global_step())

    # 计算模型准确率
    accuracy_op = tf.metrics.accuracy(labels=labels, predictions=prediction)

    # 定义estimators
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prediction,
        loss=cost,
        train_op=optimizer,
        eval_metric_ops={'accuracy': accuracy_op}
    )

    return estim_specs


model = tf.estimator.Estimator(model_fn)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True
)

model.train(input_fn=input_fn, steps=num_step)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
