import tensorflow as tf
from models.base_model import * 

FLAGS = flags.FLAGS

class MTLModel(BaseModel):
  '''Multi Task Learning'''

  def __init__(self, word_embed, semeval_data, imdb_data, is_train):
    # input data
    self.semeval_data = semeval_data
    self.imdb_data = imdb_data
    self.is_train = is_train

    # embedding initialization
    self.word_dim = word_embed.shape[1]
    w_trainable = True if self.word_dim==50 else False
    
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer=word_embed,
                                      dtype=tf.float32,
                                      trainable=w_trainable)
    pos_shape = [FLAGS.pos_num, FLAGS.pos_dim]  
    self.pos1_embed = tf.get_variable('pos1_embed', shape=pos_shape)
    self.pos2_embed = tf.get_variable('pos2_embed', shape=pos_shape)

    self.shared_layer = ConvLayer('conv_shared', FILTER_SIZES)
    self.shared_linear = LinearLayer('linear_shared', 2, True)
    with tf.variable_scope('semeval_graph'):
      self.build_semeval_graph()
    with tf.variable_scope('imdb_graph'):
      self.build_imdb_graph()

  def adversarial_loss(self, feature, label):
    '''make the task classifier cannot reliably predict the task based on 
    the shared feature
    Args:
      feature: shared feature
      label: task label
    '''
    feature = flip_gradient(feature)
    feature_size = feature.shape.as_list()[1]
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 2 classes
    logits, loss_l2 = self.shared_linear(feature)

    loss_adv = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    return loss_adv, loss_l2

  def build_semeval_graph(self):
    lexical, labels, sentence, pos1, pos2 = self.semeval_data

    # embedding lookup
    lexical = tf.nn.embedding_lookup(self.word_embed, lexical)
    lexical = tf.reshape(lexical, [-1, 6*self.word_dim])

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    # cnn model
    if self.is_train:
      sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    
    conv_layer = ConvLayer('conv_semeval', FILTER_SIZES)
    conv_out = conv_layer(sent_pos)
    conv_out = max_pool(conv_out, FLAGS.semeval_max_len)

    shared_out = self.shared_layer(sentence)
    shared_out = max_pool(shared_out, FLAGS.semeval_max_len)

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

    task_label = tf.one_hot(tf.ones_like(labels), 2)
    loss_adv, loss_adv_l2 = self.adversarial_loss(shared_out, task_label)

    self.semeval_loss = loss_ce  + FLAGS.l2_coef*loss_l2
    # self.semeval_loss = loss_ce + 0.01*loss_adv + FLAGS.l2_coef*(loss_l2+loss_adv_l2)

    self.semeval_pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(self.semeval_pred, labels), tf.float32)
    self.semeval_accuracy = tf.reduce_mean(acc)

  def build_imdb_graph(self):
    labels, sentence = self.imdb_data
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    if self.is_train:
      sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    
    conv_layer = ConvLayer('conv_imdb', FILTER_SIZES)
    conv_out = conv_layer(sentence)
    conv_out = max_pool(conv_out, FLAGS.imdb_max_len)

    shared_out = self.shared_layer(sentence)
    shared_out = max_pool(shared_out, FLAGS.imdb_max_len)

    feature = tf.concat([conv_out, shared_out], axis=1)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 2 classes
    feature_size = feature.shape.as_list()[1]
    logits, loss_l2 = linear_layer('linear_imdb_1', feature, 
                                  feature_size, FLAGS.num_imdb_class, 
                                  is_regularize=True)
    
    # xentropy= tf.nn.sigmoid_cross_entropy_with_logits(
    #                               logits=tf.squeeze(logits), 
    #                               labels=tf.cast(labels, tf.float32))
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                          labels=tf.one_hot(labels, FLAGS.num_imdb_class), 
                          logits=logits)
    loss_ce = tf.reduce_mean(xentropy)

    task_label = tf.one_hot(tf.zeros_like(labels), 2)
    loss_adv, loss_adv_l2 = self.adversarial_loss(shared_out, task_label)

    self.imdb_loss = loss_ce  + FLAGS.l2_coef*loss_l2
    # self.imdb_loss = loss_ce + 0.01*loss_adv + FLAGS.l2_coef*(loss_l2+loss_adv_l2)

    # self.imdb_pred = tf.cast(tf.greater(tf.squeeze(logits), 0.5), tf.int64)
    self.imdb_pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(self.imdb_pred, labels), tf.float32)
    self.imdb_accuracy = tf.reduce_mean(acc)

  def build_train_op(self):
    if self.is_train:
      self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
      self.semeval_train = optimize(self.semeval_loss, self.global_step)
      self.imdb_train = optimize(self.imdb_loss, self.global_step)

def optimize(loss, global_step):
  optimizer = tf.train.AdamOptimizer(FLAGS.lrn_rate)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):# for batch_norm
    train_op = optimizer.minimize(loss, global_step)
  return train_op

def build_train_valid_model(word_embed, 
                            semeval_train, semeval_test, 
                            imdb_train, imdb_test):
  with tf.name_scope("Train"):
    with tf.variable_scope('MTLModel', reuse=None):
      m_train = MTLModel(word_embed, semeval_train, imdb_train, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('MTLModel', reuse=True):
      m_valid = MTLModel(word_embed, semeval_test, imdb_test, is_train=False)
  return m_train, m_valid