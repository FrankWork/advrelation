import os
import tensorflow as tf
from tensorflow.python.framework import ops

flags = tf.app.flags
flags.DEFINE_string("logdir", "saved_models/", "where to save the model")
flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
flags.DEFINE_float("lrn_rate", 0.001, "learning rate")

FLAGS = tf.app.flags.FLAGS

FILTER_SIZES = [3, 4, 5]


class BaseModel(object):

  def set_saver(self, save_dir):
    '''
    Args:
      save_dir: relative path to FLAGS.logdir
    '''
    # shared between train and valid model instance
    self.saver = tf.train.Saver(var_list=None)
    self.save_dir = os.path.join(FLAGS.logdir, save_dir)
    self.save_path = os.path.join(self.save_dir, "model.ckpt")

  def restore(self, session):
    ckpt = tf.train.get_checkpoint_state(self.save_dir)
    self.saver.restore(session, ckpt.model_checkpoint_path)

  def save(self, session, global_step):
    self.saver.save(session, self.save_path, global_step)


class LinearLayer(tf.layers.Layer):
  '''inherit tf.layers.Layer to cache trainable variables
  '''
  def __init__(self, layer_name, in_size, out_size, is_regularize, **kwargs):
    self.layer_name = layer_name
    self.in_size = in_size
    self.out_size = out_size
    self.is_regularize = is_regularize
    super(LinearLayer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    with tf.variable_scope(self.layer_name):
      w_init = tf.truncated_normal_initializer(stddev=0.1)
      b_init = tf.constant_initializer(0.1)

      self.w = self.add_variable('W', [self.in_size, self.out_size], initializer=w_init)
      self.b = self.add_variable('b', [self.out_size], initializer=b_init)

      super(LinearLayer, self).build(input_shape)

  def call(self, x):
    loss_l2 = tf.constant(0, dtype=tf.float32)
    o = tf.nn.xw_plus_b(x, self.w, self.b)
    if self.is_regularize:
        loss_l2 += tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.b)
    return o, loss_l2


class ConvLayer(tf.layers.Layer):
  '''inherit tf.layers.Layer to cache trainable variables
  '''
  def __init__(self, layer_name, filter_sizes, **kwargs):
    self.layer_name = layer_name
    self.filter_sizes = filter_sizes
    self.conv = {} # trainable variables for conv
    super(ConvLayer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    input_dim = input_shape[2]

    with tf.variable_scope(self.layer_name):
      w_init = tf.truncated_normal_initializer(stddev=0.1)
      b_init = tf.constant_initializer(0.1)

      for fsize in self.filter_sizes:
        w_shape = [fsize, input_dim, 1, FLAGS.num_filters]
        b_shape = [FLAGS.num_filters]
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

def max_pool(conv_outs, max_len):
  pool_outs = []

  for conv in conv_outs:
    pool = tf.nn.max_pool(conv, 
                        ksize= [1, max_len, 1, 1], 
                        strides=[1, max_len, 1, 1], 
                        padding='SAME') # batch,1,1,filters
    pool_outs.append(pool)
    
  n = len(conv_outs)
  pools = tf.reshape(tf.concat(pool_outs, 3), [-1, n*FLAGS.num_filters])

  return pools

def conv_block_v2(inputs, kernel_size, num_filters, name, training, 
               batch_norm=False, initializer=None, shortcut=None, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    conv_out = tf.layers.conv1d(inputs, num_filters, kernel_size,
                  strides=1, padding='same',
                  kernel_initializer=initializer,
                  name='conv-%d' % kernel_size,
                  reuse=reuse)
    if batch_norm:
      conv_out = tf.layers.batch_normalization(conv_out, training=training)
    conv_out = tf.nn.relu(conv_out)
    if shortcut is not None:
      conv_out = conv_out + shortcut
    return conv_out
  
def optimize(loss, lrn_rate):
  optimizer = tf.train.AdamOptimizer(lrn_rate)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):# for batch_norm
    train_op = optimizer.minimize(loss)
  return train_op

class FlipGradientBuilder(object):
  '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''
  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, l=1.0):
    grad_name = "FlipGradient%d" % self.num_calls
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      return [ tf.negative(grad) * l]
    
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)
        
    self.num_calls += 1
    return y
    
flip_gradient = FlipGradientBuilder()

def scale_l2(x, norm_length=5):
  # shape(x) = (batch, num_timesteps, d)
  # Divide x by max(abs(x)) for a numerically stable L2 norm.
  # 2norm(x) = a * 2norm(x/a)
  # Scale over the full sequence, dims (1, 2)
  alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
  l2_norm = alpha * tf.sqrt(
      tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
  x_unit = x / l2_norm
  return norm_length * x_unit

def mask_by_length(t, length):
  """Mask t, 3-D [batch, time, dim], by length, 1-D [batch,]."""
  maxlen = t.get_shape().as_list()[1]

  # Subtract 1 from length to prevent the perturbation from going on 'eos'
  mask = tf.sequence_mask(length, maxlen=maxlen)
  mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
  # shape(mask) = (batch, num_timesteps, 1)
  return t * mask

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm


'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

def label_smoothing(inputs, epsilon=0.1):
  '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
  
  Args:
    inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
  '''
  K = inputs.get_shape().as_list()[-1] # number of channels
  return ((1-epsilon) * inputs) + (epsilon / K)

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs


def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        # params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
        #           "activation": tf.nn.relu, "use_bias": True}
        # outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        
        # Readout layer
        # params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
        #           "activation": None, "use_bias": True}
        # outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dense(outputs, num_units[1])
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)
    
    return outputs

def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs