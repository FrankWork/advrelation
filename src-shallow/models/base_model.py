import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("logdir", "saved_models/", "where to save the model")

FLAGS = tf.app.flags.FLAGS

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


def conv_block(inputs, input_dim, kernels_size, num_filters, name, training, 
               batch_norm=False, initializer=None, shortcut=None, reuse=None):
  inputs = tf.expand_dims(inputs, axis=-1)

  with tf.variable_scope(name, reuse=reuse):
    conv_outs = []
    for kernel_size in kernels_size:
      conv_weight = tf.get_variable('W-%d'%kernel_size, 
                              [kernel_size, input_dim, 1, num_filters],
                              initializer=initializer)
      conv_bias = tf.get_variable('b-%d'%kernel_size, [num_filters], 
                              initializer=tf.constant_initializer(0.1))
      conv_out = tf.nn.conv2d(inputs, conv_weight,
                          strides=[1, 1, input_dim, 1], padding='SAME')

      # conv_out = tf.layers.conv1d(inputs, num_filters, kernel_size,
      #               strides=input_dim, padding='same',
      #               kernel_initializer=initializer,
      #               name='conv-%d' % kernel_size,
      #               reuse=reuse)
      if batch_norm:
        conv_out = tf.layers.batch_normalization(conv_out, training=training)
      conv_out = tf.nn.relu(conv_out + conv_bias)
      if shortcut is not None:
        conv_out = conv_out + shortcut
      conv_outs.append(conv_out)
    return conv_outs


def max_pool(conv_outs, max_len, num_filters_in_conv, flat=True):
  pool_outs = []

  for conv in conv_outs:
    pool = tf.nn.max_pool(conv, ksize= [1, max_len, 1, 1], 
                              strides=[1, max_len, 1, 1], padding='SAME')
    # pool = tf.layers.max_pooling1d(conv, max_len, max_len, padding='same')
    pool_outs.append(pool)
    
  n = len(conv_outs)
  pools = tf.concat(pool_outs, 3)
  if flat:
    pools = tf.reshape(pools, [-1, n*num_filters_in_conv])
  else:
    pools = tf.squeeze(pools)

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

def pad(tensor, in_filter, out_filter):
  if out_filter != in_filter:
    n = out_filter-in_filter
    n1 = n//2
    n2 = n - n1
    return tf.pad(tensor, [[0,0], [0,0], [n1, n2]])
  return tensor
  

def optimize(loss, lrn_rate, max_norm=None, decay_steps=None):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch_norm
  with tf.control_dependencies(update_ops):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    
    if decay_steps is not None:
      lrn_rate = tf.train.exponential_decay(lrn_rate, global_step, 
                                    decay_steps, 0.95, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(lrn_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    
    if max_norm is not None:
      gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
    return train_op

  
  
