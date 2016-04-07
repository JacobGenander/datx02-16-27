import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs


class ReLURNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """ReLU RNN: output = new_state = relu(W * input + U * state + B).
       where U is initilzed to the identity matrix"""
    with vs.variable_scope(scope or type(self).__name__):  # "ReLURNNCell"

      U_init = tf.diag(tf.ones([self._num_units]))
      U = vs.get_variable("RecurrentMatrix", initializer = U_init)
      W = vs.get_variable("InputMatrix",tf.reverse(tf.shape(inputs),[True]))
      ires = tf.math_ops.matmul(inputs, W)
      sres = tf.math_ops.matmul(state, U) 
      bias_term = vs.get_variable(
        "Bias", [output_size],
        initializer=init_ops.constant_initializer(bias_start))
      res = tf.add_n([ires, sres, bias_term])
      output = tf.nn.relu(res)
    return output, output
