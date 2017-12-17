import tensorflow as tf
from models.base_model import * 

FLAGS = tf.app.flags.FLAGS

class CNNModel(BaseModel):

  def __init__(self, word_embed, data, is_train):
    # input data
    self.data = data
    self.is_train = is_train

    # embedding initialization
    self.word_dim = word_embed.shape[1]
    w_trainable = True if self.word_dim==50 else False
    
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer=word_embed,
                                      dtype=tf.float32,
                                      trainable=w_trainable)
    self.build_classify_graph()

  def build_classify_graph(self):
    labels, sentence = self.data
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    if self.is_train:
      sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    
    conv_layer = ConvLayer('conv_imdb', FILTER_SIZES)
    conv_out = conv_layer(sentence)
    conv_out = max_pool(conv_out, FLAGS.imdb_max_len)

    feature = conv_out
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 2 classes
    linear = LinearLayer('linear', 2, True)
    logits, loss_l2 = linear(feature)
    
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                          labels=tf.one_hot(labels, 2), 
                          logits=logits)
    loss_ce = tf.reduce_mean(xentropy)

    self.loss = loss_ce  + FLAGS.l2_coef*loss_l2
    
    self.pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(self.pred, labels), tf.float32)
    self.acc = tf.reduce_mean(acc)

  def build_train_op(self):
    if self.is_train:
      self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
      self.train = optimize(self.loss, self.global_step)

def build_train_valid_model(model_name, word_embed, train_data, test_data):
  with tf.name_scope("Train"):
    with tf.variable_scope(model_name, reuse=None):
      m_train = CNNModel(word_embed, train_data, is_train=True)
      m_train.set_saver(model_name)
      m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope(model_name, reuse=True):
      m_valid = CNNModel(word_embed, test_data, is_train=False)
      m_valid.set_saver(model_name)
  
  return m_train, m_valid