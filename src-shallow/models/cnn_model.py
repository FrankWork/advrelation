import tensorflow as tf
from models.base_model import * 
from models.residual import residual_net

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")
flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
flags.DEFINE_float("lrn_rate", 0.001, "learning rate")

FLAGS = flags.FLAGS

MAX_LEN = 98
NUM_CLASSES = 19
# NUM_POWER_ITER = 1
# SMALL_CONSTANT = 1e-6
KERNEL_SIZE = 3
NUM_FILTERS = 310

class CNNModel(BaseModel):

  def __init__(self, word_embed, vocab_freq, semeval_data, is_adv, is_train):
    # input data
    self.is_train = is_train
    self.is_adv = is_adv

    self.he_normal = tf.keras.initializers.he_normal()
    self.regularizer = tf.contrib.layers.l2_regularizer(FLAGS.l2_coef)

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

    # self.embed_dim = self.word_dim + 2*FLAGS.pos_dim
    self.tensors = dict()

    with tf.variable_scope('semeval_graph'):
      self.build_semeval_graph(semeval_data)

  def bottom(self, data):
    lexical, labels, length, sentence, pos1, pos2 = data

    # embedding lookup
    # weight = self.word_dim**0.5
    lexical = tf.nn.embedding_lookup(self.word_embed, lexical)
    # lexical *= weight
    # lexical = scale_l2(lexical)

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    # sentence *= weight
    # sentence = scale_l2(sentence)

    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    lexical = tf.reshape(lexical, [-1, 6*self.word_dim])

    return lexical, labels, length, sentence, pos1, pos2

  def body_shallow(self, lexical, sentence, pos1, pos2):
    # num_filters = [60, 125, 300, 60]
    num_filters = [350, 400, 450, 500]
    self.layers = []

    sentence = tf.layers.dropout(sentence, 1-FLAGS.keep_prob, training=self.is_train)
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    self.layers.append(sent_pos)

    # conv 1
    conv_input = self.layers[-1]
    conv_out = conv_block_v2(conv_input, KERNEL_SIZE, num_filters[0],
                            'conv_block1',training=self.is_train, 
                          initializer=self.he_normal, batch_norm=False)

    pool_out = tf.layers.max_pooling1d(conv_out, MAX_LEN, MAX_LEN, padding='same')
    pool_out = tf.squeeze(pool_out, axis=1)
    self.layers.append(pool_out)

  
    body_out = tf.concat([lexical, self.layers[-1]], axis=1)
    return body_out
    # return None
  
  def body(self, lexical, sentence, pos1, pos2):
    sentence = tf.layers.dropout(sentence, 1-FLAGS.keep_prob, training=self.is_train)
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)

    res_out = residual_net(sent_pos, MAX_LEN, NUM_FILTERS, self.is_train, NUM_CLASSES)
    body_out = tf.concat([lexical, res_out], axis=1)
    # # 1 conv
    # conv_input = sent_pos
    # conv_out = conv_block_v2(conv_input, KERNEL_SIZE, NUM_FILTERS,
    #                         'conv_block_final', training=self.is_train,
    #                         initializer=self.he_normal, batch_norm=False)
    # pool_out = tf.layers.max_pooling1d(conv_out, MAX_LEN, MAX_LEN, padding='same')
    # body_out = tf.squeeze(pool_out, axis=1)


    body_out = tf.layers.dropout(body_out, 1-FLAGS.keep_prob, training=self.is_train)
    return body_out
    # return None

  def top(self, body_out, labels):
    logits = tf.layers.dense(body_out, NUM_CLASSES, kernel_regularizer=self.regularizer)

    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, 
                           onehot_labels=tf.one_hot(labels, NUM_CLASSES))

      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      loss = cross_entropy + sum(regularization_losses)
      # loss = cross_entropy #+ FLAGS.l2_coef * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    return logits, loss

  def build_semeval_graph(self, data):
    lexical, labels, length, sentence, pos1, pos2 = self.bottom(data)
    body_out = self.body(lexical, sentence, pos1, pos2)
    logits, loss = self.top(body_out, labels)

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
      # self.train_op = tf.no_op()
      loss = self.tensors['loss']
      self.train_op = optimize(loss, FLAGS.lrn_rate)


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
