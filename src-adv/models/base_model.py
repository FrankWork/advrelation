import os
import tensorflow as tf
from tensorflow.python.framework import ops

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