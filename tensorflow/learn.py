#!usr/bin/env python
# -*- coding: UTF-8 -*-
'''
learn code
'''
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def session_learn():
    '''
    session
    :return:
    '''
    matrix1 = tf.constant([[3, 3]])
    matrix2 = tf.constant([[2], [2]])

    product = tf.matmul(matrix1, matrix2)

    sess = tf.Session()
    result = sess.run(product)
    print(result)

    sess.close()


def variable_learn():
    '''
    variable
    :return:
    '''

    state = tf.Variable(0, name='counter')

    one = tf.constant(1)

    val = tf.add(state, one)

    update = tf.assign(state, val)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for _ in range(3):
            sess.run(update)

        print(sess.run(state))


def placeholder_learn():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.mul(input1, input2)

    with tf.Session() as sess:
        feed = {
            input1: [2.],
            input2: [7.]
        }
        result = sess.run(output, feed_dict=feed)
        print(result)


def add_layer(inputs, in_size, out_size, activation_func=None):
    '''

    :param inputs:
    :param in_size:
    :param out_size:
    :param activation_func:
    :return:
    '''
    Weights = tf.Variable(tf.random_normal([out_size, in_size]))
    biases = tf.Variable(tf.zeros([out_size, 1]) + 0.01)

    W_x = tf.matmul(Weights, inputs) + biases

    if activation_func is None:
        outputs = W_x
    else:
        outputs = activation_func(W_x)

    return outputs


def gen_linear():
    '''
    x=[x1,
       x2,
       xn]
    :return:
    '''
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    x_holder = tf.placeholder(tf.float32, [None, 1])
    y_holder = tf.placeholder(tf.float32, [None, 1])

    l1 = add_layer(x_holder, 1, 10, activation_func=tf.nn.relu)
    prediction = add_layer(l1, 10, 1, activation_func=None)

    # sq = tf.shape(tf.reduce_sum(tf.square(y_holder - prediction),reduction_indices=[1]))

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_holder - prediction), reduction_indices=[1]))
    # loss = tf.reduce_mean(tf.square(y_holder - prediction))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    feed = {
        x_holder: x_data,
        y_holder: y_data
    }

    for i in range(101):
        sess.run(train_step, feed_dict=feed)
        if i % 10 == 0:
            val = sess.run(loss, feed_dict=feed)
            shape = sess.run(sq, feed_dict=feed)
            print('loss=', val, shape)





if '__main__' == __name__:

    if len(sys.argv) < 1:
        print 'Usage %s dict_file' % sys.argv[0]
        sys.exit(1)
    op = sys.argv[1]

    if op == 'session':
        session_learn()
    if op == 'variable':
        variable_learn()
    if op == 'placeholder':
        placeholder_learn()
    if op == 'linear':
        gen_linear()
