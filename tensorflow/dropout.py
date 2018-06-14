import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# hyper
SAMPLE_NUM = 100
HIDDEN_NUM = 300
LEARNING_RATE = 0.01

train_x = np.linspace(-1, 1, SAMPLE_NUM)[:, np.newaxis]
train_y = train_x + 0.4 * np.random.randn(SAMPLE_NUM)[:, np.newaxis]

test_x = train_x.copy()
test_y = test_x + 0.4 * np.random.randn(SAMPLE_NUM)[:, np.newaxis]

# show data
# plt.scatter(train_x, train_y, c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
# plt.legend(loc='upper left')
# plt.ylim((-2.5, 2.5))
# plt.show()

# holder
x_holder = tf.placeholder(tf.float32, [None, 1])
y_holder = tf.placeholder(tf.float32, [None, 1])
is_training = tf.placeholder(tf.bool, None)

# overfitting
over1 = tf.layers.dense(x_holder, HIDDEN_NUM, tf.nn.relu)
over2 = tf.layers.dense(over1, HIDDEN_NUM, tf.nn.relu)
over_out = tf.layers.dense(over2, 1)
over_loss = tf.losses.mean_squared_error(y_holder, over_out)
over_train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(over_loss)

# dropout
dropout1 = tf.layers.dense(x_holder, HIDDEN_NUM, tf.nn.relu)
dropout2 = tf.layers.dropout(dropout1, rate=0.5, training=is_training)
dropout3 = tf.layers.dense(dropout2, HIDDEN_NUM, tf.nn.relu)
dropout4 = tf.layers.dropout(dropout3, rate=0.5, training=is_training)
dropout_out = tf.layers.dense(dropout4, 1)

dropout_loss = tf.losses.mean_squared_error(y_holder, dropout_out)
dropout_train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(dropout_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()

for i in range(501):
    sess.run([over_train, dropout_train], feed_dict={x_holder: train_x, y_holder: train_y, is_training: True})

    if i % 20 == 0:
        plt.cla()
        over_out_result, drop_result = sess.run([over_out, dropout_out],
                                                feed_dict={x_holder: test_x, y_holder: test_y, is_training: False})

        plt.scatter(train_x, train_y, c='magenta', s=50, alpha=0.5, label='train')
        plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
        plt.plot(test_x, over_out_result, 'r-', lw=3, label='overfitting')
        plt.plot(test_x, drop_result, 'b--', lw=3, label='overfitting')

        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

plt.ioff()
plt.show()
