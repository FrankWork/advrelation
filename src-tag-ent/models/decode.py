import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

import tensorflow as tf



_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"



_TagLSTMStateTuple = collections.namedtuple("TagLSTMStateTuple", ("c", "h", "t"))

class TagLSTMStateTuple(_TagLSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h, t) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


class TagLSTMCell(tf.contrib.rnn.BasicLSTMCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
  that follows.
  """

  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None, name=None):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.

      When restoring from CudnnLSTM-trained checkpoints, must use
      `CudnnCompatibleLSTMCell` instead.
    """
    super().__init__(num_units, forget_bias=forget_bias, 
        state_is_tuple=state_is_tuple, activation=activation, reuse=reuse, 
        name=name)
    # if not state_is_tuple:
    #   logging.warn("%s: Using a concatenated state is slower and will soon be "
    #                "deprecated.  Use state_is_tuple=True.", self)

    # # Inputs must be 2-dimensional.
    # self.input_spec = base_layer.InputSpec(ndim=2)

    # self._num_units = num_units
    # self._forget_bias = forget_bias
    # self._state_is_tuple = state_is_tuple
    # self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (TagLSTMStateTuple(self._num_units, self._num_units, self._num_units)
            if self._state_is_tuple else 3 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units
    t_depth = self._num_units
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth + t_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))
    
    self._proj_layer = tf.layers.Dense(self._num_units, 
        activation=self._activation, name='proj_layer')

    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `TagLSTMStateTuple` of state tensors, each shaped
        `[batch_size, self.state_size]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size, 2 * self.state_size]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `TagLSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h, t = state
    else:
      c, h, t = array_ops.split(value=state, num_or_size_splits=3, axis=one)

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h, t], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))
    new_t = self._proj_layer(new_h)

    if self._state_is_tuple:
      new_state = TagLSTMStateTuple(new_c, new_h, new_t)
    else:
      new_state = array_ops.concat([new_c, new_h, new_t], 1)
    
    return new_t, new_state


def decode(inputs, state, lengths, hidden_size):
  # inputs = tf.transpose(inputs, [1, 0, 2]) # (max_len, batch, dim)
  lengths = tf.cast(lengths, tf.int32)

  cell = TagLSTMCell(hidden_size, name='decode_cell')
  # if state is None:
  #   state = cell.zero_state(batch_size, tf.float32)

  def initial_fn():
    zero = tf.constant(0, dtype=tf.int32)
    finished = tf.equal(zero, lengths) # all False at the initial step
    ini_input = inputs[0]
    return finished, ini_input

  def sample_fn(time, outputs, state):
    sample_ids = tf.to_int32(tf.argmax(outputs, axis=1))
    return sample_ids

  def next_inputs_fn(time, outputs, state, sample_ids):
    # next_input = tf.concat((pred_embedding, encoder_outputs[time]), 1)
    zero_input = tf.zeros_like(inputs[0])
    finished = tf.greater_equal(time, lengths-1)  # this operation produces boolean tensor of [batch_size]
    all_finished = tf.reduce_all(finished)  # -> boolean scalar
    next_inputs = tf.cond(all_finished, lambda: zero_input, lambda: inputs[time])
    next_state = state
    return finished, next_inputs, next_state

  # helper: feed encoder outputs to decoder
  helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

  decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, state)
  outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
  return outputs.rnn_output
  