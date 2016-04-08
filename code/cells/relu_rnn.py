"""
A ReLU-cell as described by http://arxiv.org/pdf/1504.00941 .
Notably the reccurent weigths are initialized to the identity
matrix times a forget factor with 1 being don't forget and 0 
being don't remember.

"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import pdb

class ReLURNNCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, num_units, forget_factor = 1):
    """ Create a ReLURNNCell with the recurrent matrix initialized to I * forget_factor.
     
     Args:
        num_units: The cell size, used both for state , input and output
        forget_factor: A factor by wich the cell will forget its current state. 
                       Default is 1 meaning that nothing is forgotten.

    """
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
       where U is initilzed to the identity matrix and the bias to 0"""
    with vs.variable_scope(scope or type(self).__name__):  # "ReLURNNCell"
      # Get or initialize the recurrent weights 
      U_init = tf.diag(tf.ones([self.output_size])) * self.forget_factor
      U = vs.get_variable("RecurrentMatrix", initializer = U_init)
      
      # Get or initialize the input weights, they are initialize uniformly 
      W = vs.get_variable("InputMatrix", shape = [self.output_size,self.output_size])

      # Get or initialize the bias, it is initialized to zero
      bias_init = tf.zeros([self.output_size])
      bias = vs.get_variable(
        "Bias", initializer=bias_init)

      # Calculate the rectified linear activation
      output = tf.nn.relu_layer(tf.concat(1,[inputs,state]),tf.concat(0, [W, U]), bias)
    return output, output
