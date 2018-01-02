import tensorflow as tf
from models.base_model import * 

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

FLAGS = flags.FLAGS

MAX_LEN = 139
CLASS_NUM = 19


class CNNModel(BaseModel):

  def __init__(self, word_embed, semeval_data, is_adv, is_train):
    # input data
    self.is_train = is_train
    self.is_adv = is_adv

    # embedding initialization
    # self.word_dim = word_embed.shape[1]
    self.word_dim = 50
    w_trainable = True #if self.word_dim==50 else False
    
    # initializer=word_embed,
    initializer= tf.random_normal_initializer(0.0, self.word_dim**-0.5)
    shape = [8097, self.word_dim]
    self.word_embed = tf.get_variable('word_embed', 
                                      shape=shape,
                                      initializer= initializer,
                                      dtype=tf.float32,
                                      trainable=w_trainable)
    pos_shape = [FLAGS.pos_num, FLAGS.pos_dim]  
    self.pos1_embed = tf.get_variable('pos1_embed', shape=pos_shape)
    self.pos2_embed = tf.get_variable('pos2_embed', shape=pos_shape)

    self.tensors = dict()

    with tf.variable_scope('semeval_graph'):
      self.build_semeval_graph(semeval_data)


  def build_semeval_graph(self, data):
    lexical, labels, sentence, pos1, pos2 = data

    # embedding lookup
    weight = self.word_dim**0.5
    lexical = tf.nn.embedding_lookup(self.word_embed, lexical)
    lexical = tf.reshape(lexical, [-1, 6*self.word_dim])
    lexical *= weight

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    sentence *= weight
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    # cnn model
    if self.is_train:
      sent_pos = tf.nn.dropout(sent_pos, FLAGS.keep_prob)
    
    conv_layer = ConvLayer('conv_semeval', FILTER_SIZES)
    conv_out = conv_layer(sent_pos)
    pool_out = max_pool(conv_out, MAX_LEN)

    feature = tf.concat([lexical, pool_out], axis=1)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 19 classes
    linear = LinearLayer('linear_semeval', CLASS_NUM, True)
    logits, loss_l2 = linear(feature)

    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                                  labels=tf.one_hot(labels, CLASS_NUM), 
                                  logits=logits)
    loss_ce = tf.reduce_mean(xentropy)

    loss = loss_ce + FLAGS.l2_coef*loss_l2

    pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(pred, labels), tf.float32)
    acc = tf.reduce_mean(acc)

    self.tensors['acc'] = acc
    self.tensors['loss'] = loss
    self.tensors['pred'] = pred


  def build_train_op(self):
    if self.is_train:
      self.train_ops = []

      loss = self.tensors['loss']
      self.train_op = optimize(loss, 0.001)


def build_train_valid_model(model_name, word_embed, 
                            semeval_train, semeval_test, 
                            is_adv, is_test):
  with tf.name_scope("Train"):
    with tf.variable_scope('CNNModel', reuse=None):
      m_train = CNNModel(word_embed, semeval_train, is_adv, is_train=True)
      m_train.set_saver(model_name)
      if not is_test:
        m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope('CNNModel', reuse=True):
      m_valid = CNNModel(word_embed, semeval_test, is_adv, is_train=False)
      m_valid.set_saver(model_name)
  return m_train, m_valid