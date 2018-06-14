import tensorflow as tf
import numpy as np

'''
W = tf.Variable([[1,2,3],[4,5,6]], dtype=tf.float32, name='Weight')
bias = tf.Variable([[1,2,3]],dtype=tf.float32, name='bias')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,'model/save.ckpt')
    print(save_path)
'''

# restore

W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='Weight')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='bias')

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'model/save.ckpt')
    print('weight=', sess.run(W))
    print('bias=', sess.run(b))
