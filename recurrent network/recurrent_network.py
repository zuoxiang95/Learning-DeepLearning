# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn

# 下载MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'F:\\', one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

logdir = r'F:\\'

# 定义输入、输出占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, n_step, n_input])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_class])

# 定义输出层权重
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_class]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_class]))
}


def RNN(inputs, weights, biases):
    # 由于输入数据是batch_size*28*28的图像，我们将其转化为[batch_size, n_step]的tensor组成的List
    x = tf.unstack(inputs, n_step, 1)

    # 定义lstm的单元
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # 获取lstm单元的输出
    outputs, states = rnn.static_rnn(lstm_cell,x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar('loss', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 定义准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# 定义初始化全局变量操作
init_op = tf.global_variables_initializer()



# 加载图
with tf.Session() as sess:
    # 初始化全局变量
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    step = 1
    # 开始训练，直到达到最大迭代次数
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # 把输入的28*28的图片修改为28条长度为28的序列
        batch_x = batch_x.reshape((batch_size, n_step, n_input))

        # 反向传播操作
        sess.run(optimizer, feed_dict={x: batch_x, y:batch_y})

        if step % display_step == 0:

            # 计算batch的准确率
            summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_x, y:batch_y})
            summary_writer.add_summary(summary, step)
            # 计算batch的损失
            summary, loss = sess.run([merged, cost], feed_dict={x: batch_x, y:batch_y})
            summary_writer.add_summary(summary, step)
            # 反向传播计算
            sess.run(optimizer, feed_dict={x: batch_x, y:batch_y})
            print(
                "Iter " + str(step * batch_size) +
                ", Minibatch Loss= " + "{:.6f}".format(loss) +
                ", Trainging Accuracy= " + "{:.5f}".format(acc)
            )
        step += 1
    print("Optimization finished!")

    # 计算测试集
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_step, n_input))
    test_label = mnist.test.labels[:test_len]
    print(
        "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y:test_label})
    )

