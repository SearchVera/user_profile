import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = 0.001
training_iter = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_class = 10

x_holder = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y_holder = tf.placeholder(tf.float32, [None, n_class])

# weight
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_class]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_class, ]))
}


def RNN(X, weights, biases):
    '''

    :param X: 128 batch, 28 steps, 28 inputs --> 128*28, 28 inputs
    :param weights:
    :param biases:
    :return:
    '''
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # 128batch, 28, 128 hidden
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    result = tf.matmul(states[1], weights['out']) + biases['out']

    return result


pred = RNN(x_holder, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_holder, logits=pred))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_holder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    step = 0
    while step * batch_size < training_iter:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_step], feed_dict={x_holder: batch_x, y_holder: batch_y})

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x_holder: batch_x, y_holder: batch_y}))

        step += 1
