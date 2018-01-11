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
NUM_FILTERS = 310

class CNNModel(BaseModel):

  def __init__(self, word_embed, semeval_data, unsup_data, is_adv, is_train):
    # input data
    self.is_train = is_train
    self.is_adv = is_adv

    self.he_normal = tf.keras.initializers.he_normal()
    self.regularizer = tf.contrib.layers.l2_regularizer(FLAGS.l2_coef)

    # embedding initialization
    self.vocab_size, self.word_dim = word_embed.shape
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer= word_embed,
                                      dtype=tf.float32,
                                      trainable=False)
    pos_shape = [FLAGS.pos_num, FLAGS.pos_dim]  
    self.pos1_embed = tf.get_variable('pos1_embed', shape=pos_shape)
    self.pos2_embed = tf.get_variable('pos2_embed', shape=pos_shape)

    self.tensors = dict()

    with tf.variable_scope('adv_graph'):
      self.build_semeval_graph(semeval_data)
      # self.build_nyt_graph(unsup_data)

  def bottom(self, data):
    labels, length, ent_pos, sentence, pos1, pos2 = data

    # mask
    pcnn_mask = self.pcnn_mask(length, ent_pos)

    # embedding lookup
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    return labels, length, pcnn_mask, sentence, pos1, pos2
  
  def conv_shallow(self, inputs):
    conv_out = conv_block_v2(inputs, KERNEL_SIZE, NUM_FILTERS,
                            'conv_block1',training=self.is_train, 
                          initializer=self.he_normal, batch_norm=False,
                          reuse=tf.AUTO_REUSE)

    pool_out = tf.layers.max_pooling1d(conv_out, MAX_LEN, MAX_LEN, padding='same')
    pool_out = tf.squeeze(pool_out, axis=1)
    return pool_out

  def pcnn_mask(self, length, ent_pos):
    n = tf.reduce_max(length)
    range = tf.expand_dims(tf.range(n), axis=0) # (1, len)

    p0 = tf.expand_dims(ent_pos[:, 0], axis=1) # (batch, 1)
    m0 = tf.less(range, p0) # (batch, len)

    p1 = tf.expand_dims(ent_pos[:, 2], axis=1) # (batch, 1)
    m1 = tf.less(p1, range) # (batch, len)
    m2 = tf.logical_not(tf.logical_or(m0, m1))

    pcnn_mask = tf.stack([m0, m1, m2], axis=-1) #(batch, len, 3)
    pcnn_mask = tf.cast(pcnn_mask, tf.float32)
    return pcnn_mask

  def pcnn(self, inputs, mask):
    conv_out = conv_block_v2(inputs, KERNEL_SIZE, NUM_FILTERS,
                            'conv_block1',training=self.is_train, 
                          initializer=self.he_normal, batch_norm=False,
                          reuse=tf.AUTO_REUSE)
    # [batch, len, d] => [batch, d, len, 1]
    conv_out = tf.expand_dims(tf.transpose(conv_out, [0, 2, 1]), axis=-1) 
    # (batch, len, 3) => [batch, 1, len, 3]
    mask = tf.expand_dims(mask, axis=1)

    pool_out = tf.reduce_max(conv_out * mask, axis=2)    # (batch, d, 3)
    pool_out = tf.reshape(pool_out, [-1, NUM_FILTERS*3]) # (batch, 3*d)

    return pool_out

  def conv_deep(self, inputs):
    # FIXME auto reuse
    return residual_net(inputs, MAX_LEN, NUM_FILTERS, self.is_train, NUM_CLASSES)

  def compute_logits(self, sentence, pos1, pos2, pcnn_mask, lexical=None, regularizer=None):
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    # conv_out = self.conv_shallow(sent_pos)
    # conv_out = self.conv_deep(sent_pos)
    conv_out = self.pcnn(sent_pos, pcnn_mask)

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
    labels, length, pcnn_mask, sentence, pos1, pos2 = self.bottom(data)
    sentence = tf.layers.dropout(sentence, FLAGS.dropout_rate, training=self.is_train)

    # cross entropy loss
    logits = self.compute_logits(sentence, pos1, pos2, pcnn_mask, regularizer=None)
    loss_xent = self.compute_xentropy_loss(logits, labels)

    # adv loss
    adv_sentence = adv_example(sentence, loss_xent)
    adv_logits = self.compute_logits(adv_sentence, pos1, pos2, pcnn_mask)
    loss_adv = self.compute_xentropy_loss(adv_logits, labels)

    # vadv loss
    loss_vadv = virtual_adversarial_loss(logits, length, sentence, pos1, pos2,
                                         pcnn_mask, self.compute_logits)

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

    self.tensors['acc'] = acc
    self.tensors['loss'] = loss_xent + loss_adv + loss_vadv #+ loss_l2
    self.tensors['pred'] = pred

  def build_nyt_graph(self, data):
    _, length, sentence, pos1, pos2 = self.bottom(data)
    sentence = tf.layers.dropout(sentence, FLAGS.dropout_rate, training=self.is_train)

    # cross entropy loss
    logits = self.compute_logits(sentence, pos1, pos2, regularizer=self.regularizer)

    # vadv loss
    loss_vadv = virtual_adversarial_loss(logits, length, sentence, pos1, pos2, self.compute_logits)

    # l2 loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_l2 = sum(regularization_losses)
   
    self.tensors['unsup_loss'] = loss_vadv #+ loss_l2

  def build_train_op(self):
    if self.is_train:
      self.train_ops = dict()
      loss = self.tensors['loss']
      self.train_ops['train_loss'] = optimize(loss, FLAGS.lrn_rate, decay_steps=None)
      # unsup_loss = self.tensors['unsup_loss']
      # self.train_ops['train_unsup_loss'] = optimize(unsup_loss, 0.1*FLAGS.lrn_rate, decay_steps=None)

def build_train_valid_model(model_name, word_embed, 
                            train_data, test_data, unsup_data,
                            is_adv, is_test):
  with tf.name_scope("Train"):
    with tf.variable_scope('CNNModel', reuse=None):
      m_train = CNNModel(word_embed, train_data, unsup_data, is_adv, is_train=True)
      m_train.set_saver(model_name)
      if not is_test:
        m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope('CNNModel', reuse=True):
      m_valid = CNNModel(word_embed, test_data, unsup_data, is_adv, is_train=False)
      m_valid.set_saver(model_name)
  return m_train, m_valid
