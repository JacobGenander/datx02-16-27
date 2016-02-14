#! /usr/bin/env python3

import tensorflow as tf
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
b = np.array([2, 2, 2])
sess = tf.Session()
op = tf.add(a,b)
res = sess.run(op)
print(res)
