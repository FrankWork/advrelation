import tensorflow as tf
from models.base_model import * 
from models.adv import *
from models.focal_loss import *
from models.residual import residual_net

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_integer("num_hops", 1, "hop numbers of entity attention")
flags.DEFINE_float("l2_coef", 0.001, "l2 loss coefficient")
flags.DEFINE_float("dropout_rate", 0.5, "dropout probability")
flags.DEFINE_float("lrn_rate", 0.001, "learning rate")

FLAGS = flags.FLAGS

MAX_LEN = 98
NUM_CLASSES = 19
KERNEL_SIZE = 3
NUM_FILTERS = 320

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

    return lexical, labels, length, sentence, pos1, pos2
  
  def conv_shallow(self, inputs):
    conv_out = conv_block_v2(inputs, KERNEL_SIZE, NUM_FILTERS,
                            'conv_block1',training=self.is_train, 
                          initializer=self.he_normal, batch_norm=False,
                          reuse=tf.AUTO_REUSE)

    pool_out = tf.layers.max_pooling1d(conv_out, MAX_LEN, MAX_LEN, padding='same')
    pool_out = tf.squeeze(pool_out, axis=1)
    return pool_out

  def conv_deep(self, inputs):
    return residual_net(inputs, MAX_LEN, NUM_FILTERS, self.is_train, NUM_CLASSES)

  def compute_logits(self, sentence, pos1, pos2, lexical=None, regularizer=None):
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    conv_out = self.conv_shallow(sent_pos)
    # conv_out = self.conv_deep(sent_pos)

    conv_out = tf.layers.dropout(conv_out, FLAGS.dropout_rate, training=self.is_train)

    logits = tf.layers.dense(conv_out, NUM_CLASSES, name='out_dense',
                        kernel_regularizer=regularizer, reuse=tf.AUTO_REUSE)
    return logits

  def compute_xentropy_loss(self, logits, labels):
    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      one_hot = tf.one_hot(labels, NUM_CLASSES)
      cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, 
                           onehot_labels=one_hot)
      # cross_entropy = tf.reduce_mean(focal_loss(one_hot, logits))

    return cross_entropy
  
  def build_semeval_graph(self, data):
    lexical, labels, length, sentence, pos1, pos2 = self.bottom(data)
    sentence = tf.layers.dropout(sentence, FLAGS.dropout_rate, training=self.is_train)

    # cross entropy loss
    logits = self.compute_logits(sentence, pos1, pos2, regularizer=self.regularizer)
    loss_xent = self.compute_xentropy_loss(logits, labels)

    # adv loss
    adv_sentence = adv_example(sentence, loss_xent)
    adv_logits = self.compute_logits(adv_sentence, pos1, pos2)
    loss_adv = self.compute_xentropy_loss(adv_logits, labels)

    # vadv loss
    loss_vadv = virtual_adversarial_loss(logits, length, sentence, pos1, pos2, self.compute_logits)

    # l2 loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_l2 = sum(regularization_losses)
    # l2_losses = []
    # for var in tf.trainable_variables():
    #   l2_losses.append(tf.nn.l2_loss(var))
    # loss_l2 = FLAGS.l2_coef*sum(l2_losses)
    
    # Accuracy
    with tf.name_scope("accuracy"):
      pred = tf.argmax(logits, axis=1)
      acc = tf.cast(tf.equal(pred, labels), tf.float32)
      acc = tf.reduce_mean(acc)

    # for t in tf.trainable_variables():
    #   print(t.op.name)
    # exit()

    self.tensors['acc'] = acc
    self.tensors['loss'] = loss_xent + loss_adv + loss_vadv + loss_l2
    self.tensors['pred'] = pred

  def build_train_op(self):
    if self.is_train:
      # self.train_op = tf.no_op()
      loss = self.tensors['loss']
      self.train_op = optimize(loss, FLAGS.lrn_rate, decay_steps=None)


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
