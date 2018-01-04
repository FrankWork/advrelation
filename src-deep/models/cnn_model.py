import tensorflow as tf
from models.base_model import * 

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_float("l2_coef", 1e-4, "l2 loss coefficient")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")
flags.DEFINE_float("lrn_rate", 1e-3, "learning rate")

FLAGS = flags.FLAGS

MAX_LEN = 98
CLASS_NUM = 19
NUM_POWER_ITER = 1
SMALL_CONSTANT = 1e-6
MAX_NORM = 7.0

he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)

class CNNModel(BaseModel):

  def __init__(self, word_embed, vocab_freq, semeval_data, is_adv, is_train):
    # input data
    self.is_train = is_train
    self.is_adv = is_adv

    # embedding initialization
    self.vocab_size, self.word_dim = word_embed.shape
    # self.word_dim = 50
    w_trainable = True if self.word_dim==50 else False
    
    initializer=word_embed
    # initializer= tf.random_normal_initializer(0.0, self.word_dim**-0.5)
    # shape = [8097, self.word_dim]
    self.word_embed = tf.get_variable('word_embed', 
                                      # shape=shape,
                                      initializer= initializer,
                                      dtype=tf.float32,
                                      trainable=w_trainable)
    # self.word_embed = self.normalize_embed(self.word_embed, vocab_freq)
    pos_shape = [FLAGS.pos_num, FLAGS.pos_dim]  
    self.pos1_embed = tf.get_variable('pos1_embed', shape=pos_shape)
    self.pos2_embed = tf.get_variable('pos2_embed', shape=pos_shape)

    self.tensors = dict()

    with tf.variable_scope('semeval_graph'):
      self.build_semeval_graph(semeval_data)

  def bottom(self, data):
    lexical, labels, length, sentence, pos1, pos2 = data

    # embedding lookup
    lexical = tf.nn.embedding_lookup(self.word_embed, lexical)
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    lexical = tf.concat(lex)
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    lexical = tf.reshape(lexical, [-1, 6*self.word_dim])

    return lexical, labels, length, sentence, pos1, pos2

  def deep_cnn(self, sentence, lexical, labels, depth=9, residual=False, pool_type='maxpool'):
    # Depth to No. Layers
    if depth == 9:
      num_layers = [2,2,2,2]
    elif depth == 17:
      num_layers = [4,4,4,4]
    elif depth == 29:
      num_layers = [10,10,4,4]
    elif depth == 49:
      num_layers = [16,16,10,6]
    else:
      raise ValueError('depth=%g is a not a valid setting!' % depth)
    
    self.layers = []

    # Temp(First) Conv Layer
    with tf.variable_scope("temp_conv") as scope: 
      filter_shape = [3, sentence.shape[-1], 64]
      W = tf.get_variable(name='W_1', shape=filter_shape, 
          initializer=he_normal,
          regularizer=regularizer)
      conv1 = tf.nn.conv1d(sentence, W, stride=1, padding="SAME")
      #conv1 = tf.nn.relu(conv1)
    print("Temp Conv", conv1.get_shape())
    self.layers.append(conv1)

    # Conv Block 64
    for i in range(num_layers[0]):
      if i < num_layers[0] - 1 and residual:
        shortcut = self.layers[-1]
      else:
        shortcut = None
      conv_block = Convolutional_Block(inputs=self.layers[-1], 
          shortcut=shortcut, num_filters=64, is_training=self.is_train, name=str(i+1))
      self.layers.append(conv_block)
    pool1 = downsampling(self.layers[-1], pool_type=pool_type, 
      name='pool1', residual=residual, shortcut=self.layers[-2])
    self.layers.append(pool1)
    print("Pooling:", pool1.get_shape())

    # Conv Block 128
    for i in range(num_layers[1]):
      if i < num_layers[1] - 1 and residual:
        shortcut = self.layers[-1]
      else:
        shortcut = None
      conv_block = Convolutional_Block(inputs=self.layers[-1], 
          shortcut=shortcut, num_filters=128, is_training=self.is_train, name=str(i+1))
      self.layers.append(conv_block)
    pool2 = downsampling(self.layers[-1], pool_type=pool_type, 
        name='pool2', residual=residual, shortcut=self.layers[-2])
    self.layers.append(pool2)
    print("Pooling:", pool2.get_shape())

    # Conv Block 256
    for i in range(num_layers[2]):
      if i < num_layers[2] - 1 and residual:
        shortcut = self.layers[-1]
      else:
        shortcut = None
      conv_block = Convolutional_Block(inputs=self.layers[-1], 
        shortcut=shortcut, num_filters=256, is_training=self.is_train, name=str(i+1))
      self.layers.append(conv_block)
    pool3 = downsampling(self.layers[-1], pool_type=pool_type, 
        name='pool3', residual=residual, shortcut=self.layers[-2])
    self.layers.append(pool3)
    print("Pooling:", pool3.get_shape())

    # Conv Block 512
    for i in range(num_layers[3]):
      if i < num_layers[3] - 1 and residual:
        shortcut = self.layers[-1]
      else:
        shortcut = None
      conv_block = Convolutional_Block(inputs=self.layers[-1], 
      shortcut=shortcut, num_filters=512, is_training=self.is_train, name=str(i+1))
      self.layers.append(conv_block)

    # Extract 8 most features as mentioned in paper
    self.k_pooled = tf.nn.top_k(tf.transpose(self.layers[-1], [0,2,1]), k=8, name='k_pool', sorted=False)[0]
    print("8-maxpooling:", self.k_pooled.get_shape())
    self.flatten = tf.reshape(self.k_pooled, (-1, 512*8))

    # fc1
    with tf.variable_scope('fc1'):
      w = tf.get_variable('w', [self.flatten.get_shape()[1], 2048], initializer=he_normal,
          regularizer=regularizer)
      b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
      out = tf.matmul(self.flatten, w) + b
      self.fc1 = tf.nn.relu(out)

    # fc2
    with tf.variable_scope('fc2'):
      w = tf.get_variable('w', [self.fc1.get_shape()[1], 2048], initializer=he_normal,
        regularizer=regularizer)
      b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
      out = tf.matmul(self.fc1, w) + b
      self.fc2 = tf.nn.relu(out)
    
    feature = tf.concat([lexical, self.fc2], axis=1)
    # fc3
    with tf.variable_scope('fc3'):
      w = tf.get_variable('w', [feature.get_shape()[1], CLASS_NUM], initializer=he_normal,
        regularizer=regularizer)
      b = tf.get_variable('b', [CLASS_NUM], initializer=tf.constant_initializer(1.0))
      self.fc3 = tf.matmul(feature, w) + b

    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3, labels=labels)
      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      loss = tf.reduce_mean(losses) + sum(regularization_losses)

    return logits, loss
  
  def build_semeval_graph(self, data):
    lexical, labels, length, sentence, pos1, pos2 = self.bottom(data)

    logits, loss = self.xentropy_logits_and_loss(lexical, sentence, pos1, pos2, labels)

    # Accuracy
    with tf.name_scope("accuracy"):
      pred = tf.argmax(logits, axis=1)
      acc = tf.cast(tf.equal(pred, labels), tf.float32)
      acc = tf.reduce_mean(acc)

    self.tensors['acc'] = acc
    self.tensors['loss'] = loss
    self.tensors['pred'] = pred

  def build_train_op(self):
    if self.is_train:
      loss = self.tensors['loss']

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.lrn_rate, global_step, 
                  FLAGS.num_epochs*FLAGS.num_batches_per_epoch, 0.95, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, MAX_NORM)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

def build_train_valid_model(model_name, word_embed, vocab_freq,
                            semeval_train, semeval_test, 
                            is_adv, is_test):
  with tf.name_scope("Train"):
    with tf.variable_scope('CNNModel', reuse=None):
      m_train = CNNModel(word_embed, vocab_freq, semeval_train, is_adv, is_train=True)
      m_train.set_saver(model_name)
      if not is_test:
        m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope('CNNModel', reuse=True):
      m_valid = CNNModel(word_embed, vocab_freq, semeval_test, is_adv, is_train=False)
      m_valid.set_saver(model_name)
  return m_train, m_valid
