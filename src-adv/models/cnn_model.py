import tensorflow as tf
from models.base_model import * 
from models.adv import *
from models.focal_loss import *
from models.residual import residual_net
from models.attention import *

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_integer("num_hops", 1, "hop numbers of entity attention")
flags.DEFINE_float("l2_coef", 0.001, "l2 loss coefficient")
flags.DEFINE_float("dropout_rate", 0.5, "dropout probability")
flags.DEFINE_float("lrn_rate", 0.001, "learning rate")

FLAGS = flags.FLAGS

MAX_LEN = 97
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
    self.embed_dim = self.word_dim + 2*FLAGS.pos_dim

    self.tensors = dict()

    with tf.variable_scope('adv_graph'):
      self.build_semeval_graph(semeval_data)
      # self.build_nyt_graph(unsup_data)

  def bottom(self, data):
    (labels, length, ent_pos, sentence, pos1, pos2) = data

    # embedding lookup
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    sentence = tf.layers.dropout(sentence, FLAGS.dropout_rate, training=self.is_train)

    return labels, length, ent_pos, sentence, pos1, pos2
  
  def conv_shallow(self, inputs, name='conv_block'):
    conv_out = conv_block_v2(inputs, KERNEL_SIZE, NUM_FILTERS,
                            name,training=self.is_train, 
                          initializer=self.he_normal, batch_norm=False,
                          reuse=tf.AUTO_REUSE)

    pool_out = tf.layers.max_pooling1d(conv_out, MAX_LEN, MAX_LEN, padding='same')
    pool_out = tf.squeeze(pool_out, axis=1)
    return pool_out

  def conv_deep(self, inputs):
    # FIXME auto reuse
    return residual_net(inputs, MAX_LEN, NUM_FILTERS, self.is_train, NUM_CLASSES)

  def compute_logits(self, sentence, length, ent_pos, pos1, pos2, regularizer=None):
    inputs = tf.concat([sentence, pos1, pos2], axis=2)

    entities = self.slice_entity(inputs, ent_pos, length)
    scaled_entities = multihead_attention(entities, inputs, None, self.embed_dim, 
                                  self.embed_dim, self.embed_dim, 10, reuse=tf.AUTO_REUSE,
                                  name='ent-mh-att')
    ent_out = tf.nn.relu(scaled_entities)
    ent_out = tf.reduce_max(ent_out, axis=1)
    # ent_out = self.conv_shallow(scaled_entities, 'conv_ent')

    conv_out = self.conv_shallow(inputs)
    # conv_out = self.conv_deep(inputs)

    out = tf.concat([ent_out, conv_out], axis=1)
    # out = conv_out
    out = tf.layers.dropout(out, FLAGS.dropout_rate, training=self.is_train)
    logits = tf.layers.dense(out, NUM_CLASSES, name='out_dense',
                        kernel_regularizer=regularizer, reuse=tf.AUTO_REUSE)
    return logits
  
  def slice_entity(self, inputs, ent_pos, length):
    '''
    Args
      conv_out: [batch, max_len, filters]
      ent_pos:  [batch, 4]
      length:   [batch]
    '''
    # slice ent1
    # -------(e1.first--e1.last)-------e2.first--e2.last-------
    begin1 = ent_pos[:, 0]
    size1 = ent_pos[:, 1] - ent_pos[:, 0] + 1

    # slice ent2
    # -------e1.first--e1.last-------(e2.first--e2.last)-------
    begin2 = ent_pos[:, 2]
    size2 = ent_pos[:, 3] - ent_pos[:, 2] + 1
    
    entities = slice_batch_n(inputs, [begin1, begin2], [size1, size2])
    entities.set_shape(tf.TensorShape([None, None, self.embed_dim]))

    return entities

  def compute_xentropy_loss(self, logits, labels):
    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      one_hot = tf.one_hot(labels, NUM_CLASSES)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                  labels=one_hot)
      # cross_entropy = tf.reduce_mean(focal_loss(one_hot, logits))

    return tf.reduce_mean(cross_entropy)
  
  def build_semeval_graph(self, data):
    labels, length, ent_pos, sentence, pos1, pos2 = self.bottom(data)

    # cross entropy loss
    logits = self.compute_logits(sentence, length, ent_pos, pos1, pos2, regularizer=self.regularizer)
    loss_xent = self.compute_xentropy_loss(logits, labels)

    # # adv loss
    adv_sentence = adv_example(sentence, loss_xent)
    adv_logits = self.compute_logits(sentence, length, ent_pos, pos1, pos2)
    loss_adv = self.compute_xentropy_loss(adv_logits, labels)

    # # vadv loss
    loss_vadv = virtual_adversarial_loss(logits, sentence, length, ent_pos, pos1, pos2, self.compute_logits)

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
    self.tensors['loss'] = loss_xent + loss_adv + loss_l2 + loss_vadv
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
