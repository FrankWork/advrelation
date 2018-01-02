from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers

import tensorflow as tf


class ConvLayer(tf.layers.Layer):
  '''inherit tf.layers.Layer to cache trainable variables
  '''
  def __init__(self, layer_name, filter_sizes=[3,4,5], num_filters=100, **kwargs):
    self.layer_name = layer_name
    self.filter_sizes = filter_sizes
    self.num_filters = num_filters
    self.conv = {} # trainable variables for conv
    super(ConvLayer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    input_dim = input_shape[2]

    with tf.variable_scope(self.layer_name):
      w_init = tf.truncated_normal_initializer(stddev=0.1)
      b_init = tf.constant_initializer(0.1)

      for fsize in self.filter_sizes:
        w_shape = [fsize, input_dim, 1, self.num_filters]
        b_shape = [self.num_filters]
        w_name = 'conv-W%d' % fsize
        b_name = 'conv-b%d' % fsize
        self.conv[w_name] = self.add_variable(
                                           w_name, w_shape, initializer=w_init)
        self.conv[b_name] = self.add_variable(
                                           b_name, b_shape, initializer=b_init)
    
      super(ConvLayer, self).build(input_shape)

  def call(self, x):
    x = tf.expand_dims(x, axis=-1)
    input_dim = x.shape.as_list()[2]
    conv_outs = []
    for fsize in self.filter_sizes:
      w_name = 'conv-W%d' % fsize
      b_name = 'conv-b%d' % fsize
      
      conv = tf.nn.conv2d(x,
                        self.conv[w_name],
                        strides=[1, 1, input_dim, 1],
                        padding='SAME')
      conv = tf.nn.relu(conv + self.conv[b_name]) # batch,max_len,1,filters
      conv_outs.append(conv)
    return conv_outs

def max_pool(conv_outs, max_len, num_filters=100):
  pool_outs = []

  for conv in conv_outs:
    pool = tf.nn.max_pool(conv, 
                        ksize= [1, max_len, 1, 1], 
                        strides=[1, max_len, 1, 1], 
                        padding='SAME') # batch,1,1,filters
    pool_outs.append(pool)
    
  n = len(conv_outs)
  pools = tf.reshape(tf.concat(pool_outs, 3), [-1, n*num_filters])

  return pools

class LinearLayer(tf.layers.Layer):
  '''inherit tf.layers.Layer to cache trainable variables
  '''
  def __init__(self, layer_name, out_size, is_regularize, **kwargs):
    self.layer_name = layer_name
    self.out_size = out_size
    self.is_regularize = is_regularize
    super(LinearLayer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    in_size = input_shape[1]

    with tf.variable_scope(self.layer_name):
      w_init = tf.truncated_normal_initializer(stddev=0.1)
      b_init = tf.constant_initializer(0.1)

      self.w = self.add_variable('W', [in_size, self.out_size], initializer=w_init)
      self.b = self.add_variable('b', [self.out_size], initializer=b_init)

      super(LinearLayer, self).build(input_shape)

  def call(self, x):
    loss_l2 = tf.constant(0, dtype=tf.float32)
    o = tf.nn.xw_plus_b(x, self.w, self.b)
    if self.is_regularize:
        loss_l2 += tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.b)
    return o, loss_l2


@registry.register_model
class RelationAdv(t2t_model.T2TModel):

  def body(self, features):
    '''
    Args:
      features: dict<string, tensor>, `inputs` tensor is the results of 
                embedding lookup of the origin `inputs` tensor. Lookup operation
                is done in `self.bottom`
    returns:
      a Tensor, shape `[batch, 1, body_output_size]`
    '''
    is_training = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
    keep_prob = 1.0-self._hparams.dropout

    inputs = common_layers.flatten4d3d(features['inputs'])
    concat1 = tf.concat([inputs, features['position1'], features['position2']], 
                       axis=-1)
    if is_training:
      concat1 = tf.nn.dropout(concat1, keep_prob)

    conv_layer = ConvLayer('conv1')
    conv_out = conv_layer(concat1)
    conv_out = max_pool(conv_out, self._hparams.max_input_seq_length)
    
    lexical = tf.reshape(features['lexical'], [-1, 6*self._hparams.hidden_size])
    concat2 = tf.concat([conv_out, lexical], axis=1)
    if is_training:
      concat2 = tf.nn.dropout(concat2, keep_prob)
   
    return tf.expand_dims(concat2, axis=1)

@registry.register_hparams
def relation_adv_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.max_input_seq_length = 138
  hparams.max_length = 138
  hparams.batch_size = 100
  hparams.use_fixed_batch_size=100
  hparams.hidden_size = 50 # word embedding dim
  hparams.dropout = 0.2
  hparams.symbol_dropout = 0.2
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 8
  hparams.kernel_height = 3
  hparams.kernel_width = 3
  hparams.learning_rate_decay_scheme = "exp50k"
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 3.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  hparams.eval_steps=80
  return hparams