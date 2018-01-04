import os
import math
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("logdir", "saved_models/", "where to save the model")
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

he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)

def Convolutional_Block(inputs, shortcut, num_filters, name, is_training):
    print("-"*20)
    print("Convolutional Block", str(num_filters), name)
    print("-"*20)

    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(1):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [3, inputs.get_shape()[2], num_filters]
                W = tf.get_variable(name='W', shape=filter_shape, 
                    initializer=he_normal,
                    regularizer=regularizer)
                out = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                out = tf.layers.batch_normalization(inputs=out, momentum=0.997, epsilon=1e-5, 
                                                center=True, scale=True, training=is_training)
                out = tf.nn.relu(out)
                print("Conv1D:", out.get_shape())
    print("-"*20)
    if shortcut is not None:
        print("-"*5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-"*5)
        return out + shortcut
    return out

# Three types of downsampling methods described by paper
def downsampling(inputs, pool_type, name, residual=False, shortcut=None):
    # k-maxpooling
    if pool_type=='k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        pool = tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=k, name=name, sorted=False)[0]
        pool = tf.transpose(pool, [0,2,1])
    # Linear
    elif pool_type=='linear':
        pool = tf.layers.conv1d(inputs=inputs, filters=inputs.get_shape()[2], kernel_size=3,
                            strides=2, padding='same', use_bias=False)
    # Maxpooling
    else:
        pool = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)
    if residual:
        shortcut = tf.layers.conv1d(inputs=shortcut, filters=shortcut.get_shape()[2], kernel_size=1,
                            strides=2, padding='same', use_bias=False)
        print("-"*5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-"*5)
        pool += shortcut
    pool = fixed_padding(inputs=pool)
    return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2]*2, kernel_size=1,
                            strides=1, padding='valid', use_bias=False)

def fixed_padding(inputs, kernel_size=3):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
    return padded_inputs
