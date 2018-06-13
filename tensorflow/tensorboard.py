import tensorflow as tf
import numpy as np

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

x_holder = tf.placeholder(tf.float32, [None, 1])
y_holder = tf.placeholder(tf.float32, [None, 1])


def add_layer(inputs, in_size, out_size, n_layer, activation_func=None):
    layer_name = 'layer%s' % n_layer

    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            # tf.summary.histogram(layer_name + '/weight', Weights)
            tf.summary.histogram('weight', Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            # tf.summary.histogram(layer_name + '/biases', biases)
            tf.summary.histogram('biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
        # tf.summary.histogram(layer_name + '/outputs', outputs)
        tf.summary.histogram('outputs', outputs)

    return outputs


l1 = add_layer(x_holder, 1, 10, n_layer=1, activation_func=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_func=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_holder - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('logs/', sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(1001):
    sess.run(train_step, feed_dict={x_holder: x_data, y_holder: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={x_holder: x_data, y_holder: y_data})
        writer.add_summary(result, i)
