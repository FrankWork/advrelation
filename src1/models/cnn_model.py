import tensorflow as tf
from models.base_model import BaseModel

flags = tf.app.flags
flags.DEFINE_integer("num_relations", 19, "number of relations")
flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")

flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("lrn_rate", 1e-3, "learning rate")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

FLAGS = flags.FLAGS

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

def conv_layer(name, input, max_len):
  with tf.variable_scope(name):
    input = tf.expand_dims(input, axis=-1)
    input_dim = input.shape.as_list()[2]

    # convolutional layer
    pool_outputs = []
    for filter_size in [3,4,5]:
      with tf.variable_scope('conv-%s' % filter_size):
        conv_weight = tf.get_variable('W1', 
                        [filter_size, input_dim, 1, FLAGS.num_filters],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_bias = tf.get_variable('b1', [FLAGS.num_filters], 
                        initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input,
                        conv_weight,
                        strides=[1, 1, input_dim, 1],
                        padding='SAME')
        conv = tf.nn.relu(conv + conv_bias) # batch_size,max_len,1,num_filters
        pool = tf.nn.max_pool(conv, 
                        ksize= [1, max_len, 1, 1], 
                        strides=[1, max_len, 1, 1], 
                        padding='SAME') # batch_size,1,1,num_filters
        pool_outputs.append(pool)
    pools = tf.reshape(tf.concat(pool_outputs, 3), [-1, 3*FLAGS.num_filters])

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

  def build_semeval_graph(self):
    lexical, rid, sentence, pos1, pos2 = self.semeval_data

    # embedding lookup
    lexical = tf.nn.embedding_lookup(self.word_embed, lexical)
    lexical = tf.reshape(lexical, [-1, 6*FLAGS.word_dim])
    labels = tf.one_hot(rid, FLAGS.num_relations)

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    # cnn model
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    if self.is_train:
      sent_pos = tf.nn.dropout(sent_pos, FLAGS.keep_prob)
    
    conv = conv_layer('conv_semeval', sent_pos, FLAGS.semeval_max_len)
    feature = tf.concat([lexical, conv], axis=1)
    feature_size = feature.shape.as_list()[1]
    
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 19 classes
    logits, loss_l2 = linear_layer('linear_semeval', feature, 
                                  feature_size, FLAGS.num_relations, 
                                  is_regularize=True)

    prediction = tf.nn.softmax(logits)
    self.semeval_prediction = tf.argmax(prediction, axis=1)

    accuracy = tf.equal(self.semeval_prediction, tf.argmax(labels, axis=1))
    self.semeval_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    loss_ce = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    self.semeval_loss = loss_ce + FLAGS.l2_coef*loss_l2

  def build_imdb_graph(self):
    label, sentence = self.imdb_train








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