"""
A ReLU-cell as described by http://arxiv.org/pdf/1504.00941 .
Notably the reccurent weigths are initialized to the identity
matrix.

"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import pdb

class ReLURNNCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, num_units, forget_factor = 1):
    self._num_units = num_units
    self._forget_factor = forget_factor

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  @property
  def forget_factor(self):
    return self._forget_factor

  def __call__(self, inputs, state, scope=None):
    """ReLU RNN: output = new_state = relu(W * input + U * state + B).
       where U is initilzed to the identity matrix"""
    with vs.variable_scope(scope or type(self).__name__):  # "ReLURNNCell"
      # Get or initialize the recurrent weights 
      U_init = tf.diag(tf.ones([self.output_size])) * self.forget_factor
      U = vs.get_variable("RecurrentMatrix", initializer = U_init)
      
      # Get or initialize the input weights, they are initialize uniformly 
      W = vs.get_variable("InputMatrix", shape = [self.output_size,self.output_size])

      # Calculate the input to the activation function
      ires = tf.matmul(inputs, W)
      sres = tf.matmul(state, U) 
      bias_term = vs.get_variable(
        "Bias", [self.output_size],
        initializer=tf.constant_initializer())
      res = ires +  sres +  bias_term

      # Calculate the rectified linear activation
      output = tf.nn.relu(res)
    return output, output