# encoding utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# build model and use mnist data-set test it
mnist = input_data.read_data_sets("E:/PycharmProjects/DL/MNIST_data/", one_hot=True)

xs = tf.placeholder(dtype=tf.float32, shape=[None, 28*28*1])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

param_detail = []


def layer_cnn(t_input):
    t_input = tf.reshape(t_input, [-1, 28, 28, 1])
    with tf.name_scope('conv-1') as scope:
        kernel = tf.Variable(
            initial_value=tf.truncated_normal(shape=[5, 5, 1, 64], dtype=tf.float32, stddev=0.01),
            name='kernel'
        )
        bias = tf.Variable(
            initial_value=tf.truncated_normal(shape=[64], dtype=tf.float32, stddev=0.01),
            name='bias'
        )
        conv1 = tf.nn.conv2d(t_input, kernel, strides=[1, 1, 1, 1], padding='SAME')
        out_1 = tf.nn.bias_add(conv1, bias)
        net = tf.nn.relu(out_1, name=scope)
        param_detail.append(net)
    with tf.name_scope('pool-1'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        param_detail.append(net)
    with tf.name_scope('conv-2') as scope:
        kernel = tf.Variable(
            initial_value=tf.truncated_normal(shape=[5, 5, 64, 64], dtype=tf.float32, stddev=0.01),
            name='kernel'
        )
        bias = tf.Variable(
            initial_value=tf.truncated_normal(shape=[64], dtype=tf.float32, stddev=0.01),
            name='bias'
        )
        conv2 = tf.nn.conv2d(net, kernel, strides=[1, 1, 1, 1], padding='SAME')
        out_2 = tf.nn.bias_add(conv2, bias)
        net = tf.nn.relu(net,name=scope)
        param_detail.append(net)
    # with tf.name_scope('pool-2') as scope:
    #     net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='VALID')
    #     param_detail.append(net)
    return net


def linear_layer(t_input):
    t_input_shape = t_input.get_shape().as_list()
    f_width = t_input_shape[1]
    f_height = t_input_shape[2]
    f_maps = t_input_shape[3]
    flatten_size = f_width * f_height * f_maps
    net = tf.reshape(t_input, shape=[-1, flatten_size])
    param_detail.append(net)
    with tf.name_scope('Linear_Layer'):
        dense_w = tf.Variable(tf.truncated_normal(shape=[flatten_size, 256],stddev=0.1), dtype=tf.float32)
        dense_b = tf.Variable(tf.truncated_normal(shape=[256]), dtype=tf.float32)
        net = tf.nn.bias_add(tf.matmul(net, dense_w), dense_b)
        param_detail.append(net)
        return net

net = layer_cnn(xs)
net = linear_layer(net)

lr = 0.001
epoch = 100000
batch_size = 128
n_input = 16
n_step = 16
n_hidden_units = 128
n_class = 10
# weights ans biases
weight = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_class]))
}
bias = {
    'in': tf.constant(0.1, shape=[n_hidden_units,]),
    'out': tf.constant(0.1, shape=[n_class,])
}


# RNN
def rnn(X, Weights, biases):
    X = tf.reshape(X, [-1, n_input])
    X_in = tf.matmul(X, weight['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_units])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(states[1], weight['out']) + biases['out']
    return results

pred = rnn(net, weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

lr = 0.001
epoch = 50
batch_size = 128
n_input = 16
n_step = 16
n_hidden_units = 128
n_class = 10
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(epoch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # batch_xs = batch_xs.reshape([batch_size, n_step, n_input])
            sess.run([train_op], feed_dict={
                xs: batch_xs,
                ys: batch_ys,
            })
            if i % 20 == 0:
                batch_Xval, batch_Yval = mnist.test.next_batch(batch_size)
                print(sess.run(accuracy, feed_dict={
                xs: batch_Xval,
                ys: batch_Yval,
            }))

for t in param_detail:
    print_activations(t)
# test ok
def tmp_test_cnn_layer():
    with tf.device('/gpu:0'):
        img = tf.reshape(x_input, shape=[-1, 28, 28, 1])
        net = layer_cnn(img)
        f_width = net.get_shape().as_list()[1]
        f_height = net.get_shape().as_list()[2]
        f_maps = net.get_shape().as_list()[3]
        flatten_shape = f_width * f_height * f_maps
        print(flatten_shape)
        net = tf.reshape(net, [-1, flatten_shape])
        param_detail.append(net)
        with tf.name_scope('dense'):
            dense_w1 = tf.Variable(tf.truncated_normal(shape=[flatten_shape, 100], stddev=0.01), dtype=tf.float32)
            dense_bias1 = tf.Variable(tf.truncated_normal(shape=[100], stddev=0.01), dtype=tf.float32)
            net = tf.matmul(net, dense_w1)
            net = tf.nn.bias_add(net, dense_bias1)
            net = tf.nn.sigmoid(net)
            param_detail.append(net)
            dense_w2 = tf.Variable(tf.truncated_normal(shape=[100, 10], stddev=0.01), dtype=tf.float32)
            dense_bias2 = tf.Variable(tf.truncated_normal(shape=[10], stddev=0.01), dtype=tf.float32)
            dense2_out = tf.matmul(net, dense_w2)
            dense_out = tf.nn.bias_add(dense2_out, dense_bias2)
            param_detail.append(net)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dense_out, labels=y_actual)
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        correct = tf.equal(tf.argmax(y_actual, axis=1), tf.argmax(dense_out, axis=1))
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch = mnist.train.next_batch(100)
            for i in range(1000):
                sess.run(train_op, feed_dict={x_input:batch[0], y_actual:batch[1]})
                if not i % 10:
                    test = mnist.train.next_batch(100)
                    print(sess.run(acc, feed_dict={x_input:test[0], y_actual:test[1]}))


