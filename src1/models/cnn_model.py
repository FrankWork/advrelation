import tensorflow as tf
from models.base_model import BaseModel

flags = tf.app.flags

flags.DEFINE_integer("num_imdb_class", 2, "number of classes for imdb labels")
flags.DEFINE_integer("num_semeval_class", 19, 
                                     "number of classes for semeval labels")
flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
flags.DEFINE_integer('hidden_size', 30,
                     'Number of hidden units in imdb classification layer.')

flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("lrn_rate", 1e-3, "learning rate")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

FLAGS = flags.FLAGS

FILTER_SIZES = [3,4,5]

def linear_layer(name, x, in_size, out_size, is_regularize=False):
  with tf.variable_scope(name):
    loss_l2 = tf.constant(0, dtype=tf.float32)
    w = tf.get_variable('linear_W', [in_size, out_size], 
                      initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('linear_b', [out_size], 
                      initializer=tf.constant_initializer(0.1))
    o = tf.nn.xw_plus_b(x, w, b) # batch_size, out_size
    if is_regularize:
      loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return o, loss_l2

class ConvLayer(tf.layers.Layer):

  def __init__(self, layer_name, filter_sizes, **kwargs):
    self.layer_name = layer_name
    self.filter_sizes = filter_sizes
    self.conv = {} # trainable variables for conv
    super(ConvLayer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    with tf.variable_scope(self.layer_name):
      w_init = tf.truncated_normal_initializer(stddev=0.1)
      b_init = tf.constant_initializer(0.1)

      for fsize in self.filter_sizes:
        w_shape = [fsize, FLAGS.word_dim, 1, FLAGS.num_filters]
        b_shape = [FLAGS.num_filters]
        w_name = 'conv-W%d' % fsize
        b_name = 'conv-b%d' % fsize
        self.conv[w_name] = self.add_variable(
                                           w_name, w_shape, initializer=w_init)
        self.conv[b_name] = self.add_variable(
                                           b_name, b_shape, initializer=b_init)
    
      super(ConvLayer, self).build(input_shape)
  
  def set_max_len(max_len):
    self.max_len = max_len

  def call(self, x):
    x = tf.expand_dims(x, axis=-1)
    pool_outputs = []

    for fsize in self.filter_sizes:
      w_name = 'conv-W%d' % fsize
      b_name = 'conv-b%d' % fsize
      
      conv = tf.nn.conv2d(x,
                        self.conv[w_name],
                        strides=[1, 1, FLAGS.word_dim, 1],
                        padding='SAME')
      conv = tf.nn.relu(conv + self.conv[b_name]) # batch,max_len,1,filters
      pool = tf.nn.max_pool(conv, 
                        ksize= [1, self.max_len, 1, 1], 
                        strides=[1, self.max_len, 1, 1], 
                        padding='SAME') # batch,1,1,filters
      pool_outputs.append(pool)
    
    n = len(self.filter_sizes)
    pools = tf.reshape(tf.concat(pool_outputs, 3), [-1, n*FLAGS.num_filters])

    return pools

class MTLModel(BaseModel):
  '''Multi Task Learning'''

  def __init__(self, word_embed, semeval_data, imdb_data, is_train):
    # input data
    self.semeval_data = semeval_data
    self.imdb_data = imdb_data
    self.is_train

    # embedding initialization
    w_trainable = True if FLAGS.word_dim==50 else False
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer=word_embed,
                                      dtype=tf.float32,
                                      trainable=w_trainable)
    pos_shape = [FLAGS.pos_num, FLAGS.pos_dim]  
    self.pos1_embed = tf.get_variable('pos1_embed', shape=pos_shape)
    self.pos2_embed = tf.get_variable('pos2_embed', shape=pos_shape)

    self.shared_layer = ConvLayer('conv_shared', FILTER_SIZES)
    self.build_semeval_graph()
    self.build_semeval_graph()

  def build_semeval_graph(self):
    lexical, labels, sentence, pos1, pos2 = self.semeval_data

    # embedding lookup
    lexical = tf.nn.embedding_lookup(self.word_embed, lexical)
    lexical = tf.reshape(lexical, [-1, 6*FLAGS.word_dim])

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    # cnn model
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    if self.is_train:
      sent_pos = tf.nn.dropout(sent_pos, FLAGS.keep_prob)
    
    conv_layer = ConvLayer('conv_semeval', FILTER_SIZES)
    conv_layer.set_max_len(FLAGS.semeval_max_len)
    conv_out = conv_layer(sent_pos)

    self.shared_layer.set_max_len(FLAGS.semeval_max_len)
    shared_out = self.shared_layer(sent_pos)

    feature = tf.concat([lexical, conv_out, shared_out], axis=1)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 19 classes
    feature_size = feature.shape.as_list()[1]
    logits, loss_l2 = linear_layer('linear_semeval', 
                                  feature, 
                                  feature_size, 
                                  FLAGS.num_semeval_class, 
                                  is_regularize=True)

    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                          labels=tf.one_hot(labels, FLAGS.num_semeval_class), 
                          logits=logits)
    loss_ce = tf.reduce_mean(xentropy)
    self.semeval_loss = loss_ce + FLAGS.l2_coef*loss_l2

    self.semeval_pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(self.semeval_pred, labels), tf.float32)
    self.semeval_accuracy = tf.reduce_mean(acc)

  def build_imdb_graph(self):
    labels, sentence = self.imdb_train
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    if self.is_train:
      sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    
    conv_layer = ConvLayer('conv_imdb', FILTER_SIZES)
    conv_layer.set_max_len(FLAGS.imdb_max_len)
    conv_out = conv_layer(sentence)

    self.shared_layer.set_max_len(FLAGS.imdb_max_len)
    shared_out = self.shared_layer(sentence)

    feature = tf.concat([conv_out, shared_out], axis=1)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 2 classes
    feature_size = feature.shape.as_list()[1]
    logits, loss_l2 = linear_layer('linear_imdb_1', feature, 
                                  feature_size, FLAGS.hidden_size, 
                                  is_regularize=True)
    logits, _ = linear_layer('linear_imdb_2', logits, 
                                  logits.shape.as_list()[1], 1, 
                                  is_regularize=False)
    
    xentropy= tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=tf.squeeze(logits), 
                                  labels=tf.cast(labels, tf.float32))
    loss_ce = tf.reduce_mean(xentropy)
    self.imdb_loss = loss_ce + FLAGS.l2_coef*loss_l2

    self.imdb_pred = tf.cast(tf.greater(tf.squeeze(logits), 0.5), tf.int64)
    acc = tf.cast(tf.equal(self.semeval_pred, labels), tf.float32)
    self.imdb_accuracy = tf.reduce_mean(acc)

  def set_shared_max_len(self, max_len):
    self.shared_layer.set_max_len(max_len)

def optimize():
  global_step = tf.Variable(0, trainable=False, name='step', dtype=tf.int32)
  optimizer = tf.train.AdamOptimizer(lrn_rate)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):# for batch_norm
    self.train_op = optimizer.minimize(self.loss, global_step)
  self.global_step = global_step

def build_train_valid_model(word_embed, train_data, test_data):
  '''Relation Classification via Convolutional Deep Neural Network'''
  with tf.name_scope("Train"):
    with tf.variable_scope('CNNModel', reuse=None):
      m_train = CNNModel( word_embed, train_data, FLAGS.word_dim,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    FLAGS.keep_prob, FLAGS.num_filters, 
                    FLAGS.lrn_rate, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('CNNModel', reuse=True):
      m_valid = CNNModel( word_embed, test_data, FLAGS.word_dim,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    1.0, FLAGS.num_filters, 
                    FLAGS.lrn_rate, is_train=False)
  return m_train, m_valid