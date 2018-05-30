#!usr/bin/env python
# -*- coding: UTF-8 -*-
'''
线性拟合
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np

# 准备数据
x = np.random.rand(100).astype(np.float32)
y = x * 0.1 + 0.3

## 构建结构
learning_rate = 0.5
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biase = tf.Variable(tf.zeros([1]))

y_pred = weight * x + biase

loss = tf.reduce_mean(tf.square(y - y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

##训练
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weight), sess.run(biase), sep='\t')
