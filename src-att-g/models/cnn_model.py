import tensorflow as tf
from models.base_model import * 
from models.attention import *
from models.residual import residual_net

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_integer("num_hops", 1, "hop numbers of entity attention")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("dropout_rate", 0.5, "dropout probability")
flags.DEFINE_float("lrn_rate", 0.001, "learning rate")

FLAGS = flags.FLAGS

MAX_LEN = 97
NUM_CLASSES = 19
# NUM_POWER_ITER = 1
# SMALL_CONSTANT = 1e-6
KERNEL_SIZE = 3
NUM_FILTERS = 310

class CNNModel(BaseModel):

  def __init__(self, word_embed, semeval_data, is_adv, is_train):
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
    self.cache = list()

    # with tf.variable_scope('semeval_graph'):
    self.build_semeval_graph(semeval_data)

  def body_v0(self, data):
    (label, length, ent_pos, sentence, pos1, pos2) = data

    # sentence and pos from embedding
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    # conv
    sentence = tf.layers.dropout(sentence, FLAGS.dropout_rate, training=self.is_train)
    inputs = tf.concat([sentence, pos1, pos2], axis=2)
    # self.cache.append(inputs)

    entities = self.slice_entity(inputs, ent_pos, length)
    scaled_entities = multihead_attention(entities, inputs, None, self.embed_dim, 
                                  self.embed_dim, self.embed_dim, 10, 
                                  name='ent-mh-att')
    conv_ent = conv_block_v2(scaled_entities, KERNEL_SIZE, NUM_FILTERS,
                            'conv_block1',training=self.is_train, 
                             initializer=self.he_normal, reuse=tf.AUTO_REUSE)
    pool_ent = tf.layers.max_pooling1d(conv_ent, MAX_LEN, MAX_LEN, padding='same')
    pool_ent = tf.squeeze(pool_ent, axis=1)

    conv_out = conv_block_v2(inputs, KERNEL_SIZE, NUM_FILTERS,
                            'conv_block2',training=self.is_train, 
                             initializer=self.he_normal, reuse=tf.AUTO_REUSE)
    # pool_out = tf.layers.max_pooling1d(conv_out, MAX_LEN, MAX_LEN, padding='same')
    # pool_out = tf.squeeze(pool_out, axis=1)
    pool_max = tf.reduce_max(conv_out, axis=1)
    # self.cache.append(conv_out)

    # ent1, ent2, context = self.slice_ent_and_context(conv_out, ent_pos, length)
    # pool_att = self.entity_attention(context, ent1, ent2)

    # pool_att = self.entity_attention_v2(conv_out, pool_ent)

    pool_out = tf.concat([pool_ent, pool_max], axis=1)
    # pool_out = pool_out1

    body_out = tf.layers.dropout(pool_out, FLAGS.dropout_rate, training=self.is_train)
    return label, body_out

  def body(self, data):
    (label, length, ent_pos, sentence, pos1, pos2) = data

    # sentence and pos from embedding
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    # conv
    sentence = tf.layers.dropout(sentence, FLAGS.dropout_rate, training=self.is_train)
    inputs = tf.concat([sentence, pos1, pos2], axis=2)
    # self.cache.append(inputs)

    entities = self.slice_entity(inputs, ent_pos, length)
    scaled_entities = multihead_attention(entities, inputs, None, self.embed_dim, 
                                  self.embed_dim, self.embed_dim, 10, 
                                  name='ent-mh-att')
    conv_ent = conv_block_v2(scaled_entities, KERNEL_SIZE, NUM_FILTERS,
                            'conv_block1',training=self.is_train, 
                             initializer=self.he_normal, reuse=tf.AUTO_REUSE)
    pool_ent = tf.layers.max_pooling1d(conv_ent, MAX_LEN, MAX_LEN, padding='same')
    # pool_ent = tf.squeeze(pool_ent, axis=1)

    inp_len = tf.shape(inputs)[1]
    ent_tile = tf.tile(pool_ent, [1, inp_len, 1])
    inputs = tf.concat([inputs, ent_tile], axis=2)

    conv_out = conv_block_v2(inputs, KERNEL_SIZE, 2*NUM_FILTERS,
                            'conv_block2', training=self.is_train, 
                             initializer=self.he_normal, reuse=tf.AUTO_REUSE)
    pool_max = tf.reduce_max(conv_out, axis=1)
    # self.cache.append(conv_out)
    # pool_out = tf.concat([pool_ent, pool_max], axis=1)
    pool_out = pool_max

    body_out = tf.layers.dropout(pool_out, FLAGS.dropout_rate, training=self.is_train)
    return label, body_out

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

  def slice_ent_and_context(self, conv_out, ent_pos, length):
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
    ent1 = slice_batch(conv_out, begin1, size1)

    # slice ent2
    # -------e1.first--e1.last-------(e2.first--e2.last)-------
    begin2 = ent_pos[:, 2]
    size2 = ent_pos[:, 3] - ent_pos[:, 2] + 1
    ent2 = slice_batch(conv_out, begin2, size2)
    
    # slice context
    # (-------)e1.first--e1.last(-------)e2.first--e2.last(-------)
    size1 = ent_pos[:, 0]
    begin1 = tf.zeros_like(size1, dtype=tf.int32)

    begin2 = ent_pos[:, 1]+1
    size2 = ent_pos[:, 2] - ent_pos[:, 1] - 1

    begin3 = ent_pos[:, 3]+1
    size3 = length-ent_pos[:, 3]-1

    context = slice_batch_n(conv_out, [begin1, begin2, begin3], [size1, size2, size3])

    ent1.set_shape(tf.TensorShape([None, None, self.embed_dim]))
    ent2.set_shape(tf.TensorShape([None, None, self.embed_dim]))
    context.set_shape(tf.TensorShape([None, None, self.embed_dim]))

    return ent1, ent2, context
    
  def conv_shallow(self, inputs):
    conv_out = conv_block_v2(inputs, KERNEL_SIZE, NUM_FILTERS,
                            'conv_block1',training=self.is_train, 
                          initializer=self.he_normal, batch_norm=False)

    pool_out = tf.layers.max_pooling1d(conv_out, MAX_LEN, MAX_LEN, padding='same')
    pool_out = tf.squeeze(pool_out, axis=1)
    return pool_out

  def conv_deep(self, inputs):
    return residual_net(inputs, MAX_LEN, NUM_FILTERS, self.is_train)

  def entity_attention(self, context, ent1, ent2, num_hops=1):
    def tanh_att(ent, context, name, keepdims=True):
      cont_len = tf.shape(context)[1]
      ent_tile = tf.tile(ent, [1, cont_len, 1])
      inputs = tf.concat([context, ent_tile], axis=2)
      weight = tf.layers.dense(inputs, 1, activation=tf.nn.tanh, 
                               kernel_regularizer=None, 
                               name=name, reuse=tf.AUTO_REUSE)
      weight = tf.nn.softmax(weight, axis=1)
      att_out = tf.multiply(context, weight)
      return tf.reduce_sum(att_out, axis=1, keepdims=keepdims)

    ent1 = tf.reduce_mean(ent1, axis=1, keepdims=True)
    ent2 = tf.reduce_mean(ent2, axis=1, keepdims=True)
    ent = tf.concat([ent1, ent2], axis=2)

    out = tanh_att(ent, context, 'att', keepdims=False)

    return out

  def entity_attention_v1(self, context, entities, num_hops=1):
    def tanh_att(ent, context, name, keepdims=True):
      cont_len = tf.shape(context)[1]
      ent_tile = tf.tile(ent, [1, cont_len, 1])
      inputs = tf.concat([context, ent_tile], axis=2)
      weight = tf.layers.dense(inputs, 1, activation=tf.nn.tanh, 
                               kernel_regularizer=None, 
                               name=name, reuse=tf.AUTO_REUSE)
      weight = tf.nn.softmax(weight, axis=1)
      att_out = tf.multiply(context, weight)
      return tf.reduce_sum(att_out, axis=1, keepdims=keepdims)

    ent = tf.reduce_mean(entities, axis=1, keepdims=True)
    
    out = tanh_att(ent, context, 'att', keepdims=False)
    return out

  def entity_attention_v2(self, context, entities, num_hops=1):
    def tanh_att(ent, context, name, keepdims=True):
      cont_len = tf.shape(context)[1]
      ent_tile = tf.tile(ent, [1, cont_len, 1])
      inputs = tf.concat([context, ent_tile], axis=2)
      weight = tf.layers.dense(inputs, 1, activation=tf.nn.tanh, 
                               kernel_regularizer=None, 
                               name=name, reuse=tf.AUTO_REUSE)
      weight = tf.nn.softmax(weight, axis=1)
      att_out = tf.multiply(context, weight)
      return tf.reduce_sum(att_out, axis=1, keepdims=keepdims)

    ent = tf.expand_dims(entities, axis=1)
    
    out = tanh_att(ent, context, 'att', keepdims=False)
    return out

  def attentive_pool(self, conv_out):
    h1_cnn = conv_out
    max_len = tf.shape(conv_out)[1]
    ## Attentive pooling
    W_a1 = tf.get_variable("W_a1", shape=[NUM_FILTERS, NUM_FILTERS])        # 100x100
    tmp1 = tf.matmul(tf.reshape(h1_cnn, shape=[-1, NUM_FILTERS]), W_a1, name="Wy")   # NMx100
    h2_cnn = tf.reshape(tmp1, shape=[-1, max_len, NUM_FILTERS])         #NxMx100

    M = tf.tanh(h2_cnn)                  # NxMx100
    W_a2 = tf.get_variable("W_a2", shape=[NUM_FILTERS, 1])         # 100 x 1
    tmp3 = tf.matmul(tf.reshape(M, shape=[-1, NUM_FILTERS]), W_a2)      # NMx1
    alpha = tf.nn.softmax(tf.reshape(tmp3, shape=[-1, max_len], name="att"))  # NxM  
    self.ret_alpha = alpha

    alpha = tf.expand_dims(alpha, 1)             # Nx1xM
    h2_pool = tf.matmul(alpha, tf.reshape(h1_cnn, shape=[-1, max_len, NUM_FILTERS]), name="r")
    h2_pool = tf.squeeze(h2_pool, axis=1)
    return h2_pool

  def top(self, body_out, labels):
    logits = tf.layers.dense(body_out, NUM_CLASSES, kernel_regularizer=self.regularizer)

    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      one_hot = tf.one_hot(labels, NUM_CLASSES)
      # one_hot = label_smoothing(one_hot)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot,
                                logits=logits)

      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      loss = tf.reduce_mean(cross_entropy) + sum(regularization_losses)
      # l2_losess = [0.001*tf.nn.l2_loss(t) for t in tf.trainable_variables()]
      # loss = tf.reduce_mean(cross_entropy) + tf.add_n(l2_losess)

    return logits, loss

  def build_semeval_graph(self, data):
    labels, body_out = self.body(data)
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