import tensorflow as tf
from models.base_model import * 

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

FLAGS = flags.FLAGS

MAX_LEN = 98
CLASS_NUM = 19


class CNNModel(BaseModel):

  def __init__(self, word_embed, semeval_data, is_adv, is_train):
    # input data
    self.is_train = is_train
    self.is_adv = is_adv

    # embedding initialization
    self.word_dim = word_embed.shape[1]
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
    pos_shape = [FLAGS.pos_num, FLAGS.pos_dim]  
    self.pos1_embed = tf.get_variable('pos1_embed', shape=pos_shape)
    self.pos2_embed = tf.get_variable('pos2_embed', shape=pos_shape)

    self.tensors = dict()

    self.conv_layer = ConvLayer('conv_semeval', FILTER_SIZES)
    self.linear_layer = LinearLayer('linear_semeval', CLASS_NUM, True)

    with tf.variable_scope('semeval_graph'):
      self.build_semeval_graph(semeval_data)

  def bottom(self, data):
    lexical, labels, sentence, pos1, pos2 = data

    # embedding lookup
    weight = self.word_dim**0.5
    lexical = tf.nn.embedding_lookup(self.word_embed, lexical)
    # lexical *= weight

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    # sentence *= weight
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    # cnn model
    if self.is_train:
      sent_pos = tf.nn.dropout(sent_pos, FLAGS.keep_prob)
    
    return lexical, labels, sent_pos

  def xentropy_loss(self, lexical, sent_pos, labels, l2_coef=0.01):
    conv_out = self.conv_layer(sent_pos)
    pool_out = max_pool(conv_out, MAX_LEN)

    lexical = tf.reshape(lexical, [-1, 6*self.word_dim])
    feature = tf.concat([lexical, pool_out], axis=1)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 19 classes
    logits, loss_l2 = self.linear_layer(feature)

    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                                  labels=tf.one_hot(labels, CLASS_NUM), 
                                  logits=logits)
    loss_ce = tf.reduce_mean(xentropy)

    loss = loss_ce + l2_coef*loss_l2

    return logits, loss
  
  def adv_example(self, input, loss):
    grad, = tf.gradients(
        loss,
        input,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = scale_l2(grad)
    return input + perturb

  def adversarial_loss(self, lexical, sent_pos, loss, labels):
    adv_lexical = self.adv_example(lexical, loss)
    adv_sent_pos = self.adv_example(sent_pos, loss)
    _, loss = self.xentropy_loss(adv_lexical, adv_sent_pos, labels, l2_coef=0)
    return loss

  def build_semeval_graph(self, data):
    lexical, labels, sent_pos = self.bottom(data)
    
    logits, loss = self.xentropy_loss(lexical, sent_pos, labels)
    adv_loss = self.adversarial_loss(lexical, sent_pos, loss, labels)

    pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(pred, labels), tf.float32)
    acc = tf.reduce_mean(acc)

    self.tensors['acc'] = acc
    self.tensors['loss'] = loss + adv_loss
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