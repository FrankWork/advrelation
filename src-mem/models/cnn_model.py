import tensorflow as tf
from models.base_model import * 

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

FLAGS = flags.FLAGS

MAX_LEN = 98
NUM_CLASSES = 19
NUM_POWER_ITER = 1
SMALL_CONSTANT = 1e-6

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

    self.embed_dim = self.word_dim + 2* FLAGS.pos_dim

    self.tensors = dict()

    with tf.variable_scope('semeval_graph'):
      self.build_semeval_graph(semeval_data)

  def bottom(self, data):
    (label, length, 
      sentence, position1, position2, 
      ent1_toks, ent1_pos1, ent1_pos2,
      ent2_toks, ent2_pos1, ent2_pos2,
      context, cont_pos1, cont_pos2) = data

    self.context = context

    # sentence and pos from embedding
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    pos1 = tf.nn.embedding_lookup(self.pos1_embed, position1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, position2)

    # entities, context and pos from memory
    ent1_toks = tf.nn.embedding_lookup(self.word_embed, ent1_toks)
    ent2_toks = tf.nn.embedding_lookup(self.word_embed, ent2_toks)
    context = tf.nn.embedding_lookup(self.word_embed, context)

    ent1_pos1 = tf.nn.embedding_lookup(self.pos1_embed, ent1_pos1)
    ent1_pos2 = tf.nn.embedding_lookup(self.pos2_embed, ent1_pos2)
    ent2_pos1 = tf.nn.embedding_lookup(self.pos1_embed, ent2_pos1)
    ent2_pos2 = tf.nn.embedding_lookup(self.pos2_embed, ent2_pos2)
    cont_pos1 = tf.nn.embedding_lookup(self.pos1_embed, cont_pos1)
    cont_pos2 = tf.nn.embedding_lookup(self.pos2_embed, cont_pos2)

    # concat pos embedding to entities and context
    # context = tf.concat([context, cont_pos1, cont_pos2], axis=2)
    # ent1 = tf.concat([ent1_toks, ent1_pos1, ent1_pos2], axis=2)
    # ent2 = tf.concat([ent2_toks, ent2_pos1, ent2_pos2], axis=2)

    ent1 = ent1_toks
    ent2 = ent2_toks

    batch = 100
    hop = 3

    ########################################
    # input attention
    ########################################
    ent1 = tf.reduce_mean(ent1, axis=1)
    ent2 = tf.reduce_mean(ent2, axis=1)
    
    def attention(ent, context):
      A1 = tf.matmul(context, tf.expand_dims(ent, -1))# bz, n, 1
      A1 = tf.squeeze(A1, axis=-1)
      alpha1 = tf.nn.softmax(A1)# bz, n
      
      return tf.expand_dims(alpha1, -1)
      
    alpha1 = attention(ent1, context)
    ent1 = tf.multiply(context, alpha1) # bz, n, dc
    ent1 = tf.reduce_mean(ent1, axis=1)

    alpha2 = attention(ent2, context)
    ent2 = tf.multiply(context, alpha2) # bz, n, dc
    ent2 = tf.reduce_mean(ent2, axis=1)

    self.alpha1 = tf.squeeze(alpha1, axis = -1)
    self.alpha2 = tf.squeeze(alpha2, axis = -1)
    

    entities = tf.concat([ent1, ent2], axis=-1)
    ########################################
    # multi-head attention
    ########################################

    # self.orig_ent1 = ent1
    # self.orig_ent2 = ent2
    # self.context = context

    # ent1 = multihead_attention(context, ent1, num_heads=10, scope='att1-0', 
    #                                     is_training=self.is_train)#, reuse=tf.AUTO_REUSE)
    # ent2 = multihead_attention(context, ent2, num_heads=10, scope='att2-0',
    #                                     is_training=self.is_train)#, reuse=tf.AUTO_REUSE)

    # self.ent1 = ent1
    # self.ent2 = ent2

    # for i in range(1, hop):
    #   ent1 = multihead_attention(context, ent1, num_heads=10, 
    #                                             scope='att1%d'%i)#, reuse=tf.AUTO_REUSE)
    #   ent2 = multihead_attention(context, ent2, num_heads=10, 
    #                                             scope='att2%d'%i)#, reuse=tf.AUTO_REUSE)
    # ent1 = tf.reduce_max(ent1, axis=1) # (batch, 1, embed)
    # ent2 = tf.reduce_max(ent2, axis=1)
    # entities = tf.concat([ent1, ent2], axis=-1)
    ## entities = tf.squeeze(entities, axis=1)


    # cont1 = context
    # cont2 = context
    # for i in range(hop):
    #   cont1 = multihead_attention(cont1, ent1, num_heads=10, 
    #                                 is_training=self.is_train, scope='att1%d'%i)
    #   cont2 = multihead_attention(cont2, ent2, num_heads=10, 
    #                                 is_training=self.is_train, scope='att2%d'%i)
    #   cont1 = feedforward(cont1, num_units=[620, 310], scope='ffd1%d'%i)
    #   cont2 = feedforward(cont2, num_units=[620, 310], scope='ffd2%d'%i)
    #   ent1 = cont1
    #   ent2 = cont2

    # ent1 = tf.reduce_max(ent1, axis=1) # (batch, 1, embed)
    # ent2 = tf.reduce_max(ent2, axis=1)
    # entities = tf.concat([ent1, ent2], axis=-1)
    # entities = tf.squeeze(entities, axis=1)
    
    return entities, label

  def top(self, body_out, labels):
    logits = tf.layers.dense(body_out, NUM_CLASSES, kernel_regularizer=self.regularizer)

    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      one_hot = tf.one_hot(labels, NUM_CLASSES)
      # one_hot = label_smoothing(one_hot)
      cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, 
                           onehot_labels=one_hot)

      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      loss = cross_entropy + sum(regularization_losses)

    return logits, loss

  def build_semeval_graph(self, data):
    body_out, labels = self.bottom(data)
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
      self.train_ops = []

      loss = self.tensors['loss']
      self.train_op = optimize(loss, 0.001)


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