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

    # process entities and context
    context = tf.concat([context, cont_pos1, cont_pos2], axis=2)
    ent1 = tf.concat([ent1_toks, ent1_pos1, ent1_pos2], axis=2)
    ent2 = tf.concat([ent2_toks, ent2_pos1, ent2_pos2], axis=2)
    # ent1 = ent1_toks
    # ent2 = ent2_toks

    batch = 100
    hop = 3

    
    ########################################
    # reduce max attention
    ########################################
    ent1 = tf.reduce_max(ent1, axis=1) # (batch, 1, embed)
    ent2 = tf.reduce_max(ent2, axis=1)
    entities = tf.concat([ent1, ent2], axis=-1) # (batch, 2*embed)
    
    
    ########################################
    # multi-head attention
    ########################################
    # from tensor2tensor.layers import common_attention
    # padding = common_attention.embedding_to_padding(x)
    # attention_bias = common_attention.attention_bias_ignore_padding(padding)
    # x = common_layers.layer_prepostprocess(
    #                 None,
    #                 x,
    #                 sequence='n',
    #                 dropout_rate=0.1,
    #                 norm_type='layer',
    #                 depth=None,
    #                 epsilon=1e-6,
    #                 default_name="layer_prepostprocess")
    # y = common_attention.multihead_attention(
    #       query_antecedent = x,
    #       memory_antecedent = None,
    #       bias= attention_bias,
    #       total_key_depth = self.embed_dim,
    #       total_value_depth = self.embed_dim,
    #       output_depth = self.embed_dim,
    #       num_heads = 8,
    #       dropout_rate = 0.0,
    #       attention_type="dot_product")


    # ent1 = tf.reduce_mean(ent1, axis=1, keep_dims=True) # (batch, 1, embed)
    # ent2 = tf.reduce_mean(ent2, axis=1, keep_dims=True)

    # hop = 1
    # for i in range(hop):
    #   ent1_val = multihead_attention(context, ent1, num_heads=10, 
    #                                             scope='att1%d'%i)#, reuse=tf.AUTO_REUSE)
    #   ent2_val = multihead_attention(context, ent2, num_heads=10, 
    #                                             scope='att2%d'%i)#, reuse=tf.AUTO_REUSE)

    #   ent1 = tf.reduce_mean(ent1_val, axis=1, keep_dims=True) # (batch, 1, embed)
    #   ent2 = tf.reduce_mean(ent2_val, axis=1, keep_dims=True)

    # entities = tf.concat([ent1, ent2], axis=-1)
    # entities = tf.squeeze(entities, axis=1)
    

    ########################################
    # input attention
    ########################################
    # def attention_fn(x, ent):
    #   ''' dot product attention
    #   Args
    #     x:   [batch, len, d]
    #     ent: [batch, d] => [batch, d, 1]
    #   Returns:
    #     value: [batch, d]
    #   '''
    #   logits = tf.matmul(x, tf.expand_dims(ent, -1))# [batch, len, 1]
    #   weights = tf.nn.softmax(logits, dim=1)
    #   weighted_x = tf.multiply(x, weights) # [batch, len, d]
    #   return tf.reduce_sum(weighted_x, axis=1) # [batch, d]
    
    # for _ in range(hop):
    #   ent1 = attention_fn(context, ent1)
    #   ent2 = attention_fn(context, ent2)
      
    #   ent1_proj = tf.layers.dense(ent1, self.embed_dim)
    #   ent2_proj = tf.layers.dense(ent2, self.embed_dim)

    #   ent1 = ent1 + ent1_proj
    #   ent2 = ent2 + ent1_proj

    # entities = tf.concat([ent1, ent2], axis=-1)


    ########################################
    # conv attention
    ########################################
    # ent1 = conv_block_v2(ent1, 3, 310, 'conv-ent1', training=self.is_train)
    # ent1 = tf.layers.max_pooling1d(ent1, MAX_LEN, MAX_LEN, padding='same')

    # ent2 = conv_block_v2(ent2, 3, 310, 'conv-ent2', training=self.is_train)
    # ent2 = tf.layers.max_pooling1d(ent2, MAX_LEN, MAX_LEN, padding='same')

    # entities = tf.concat([ent1, ent2], axis=-1)
    # entities = tf.squeeze(entities, axis=1)


    
    ########################################
    # word attention
    ########################################
    # self.w_att = tf.get_variable('w_att', [3*self.embed_dim, 1])
    # self.b_att = tf.get_variable('b_att', [1])

    # shape = tf.ones_like(tf.concat([context, context], axis=-1))
    
    # for _ in range(hop):
    #   e3d = tf.expand_dims(entities, axis=1) # (batch, 1, 2*d)
    #   e_tile = shape * e3d
    #   x3d = tf.concat([e_tile, context], axis=-1) # (batch, len, 3d)

    #   w3d = tf.tile(tf.expand_dims(self.w_att, axis=0), [batch, 1, 1])# (batch, 3d, 1)
    #   g = tf.nn.tanh(tf.nn.xw_plus_b(x3d, w3d, self.b_att)) #(batch, len, 1)

    #   alpha = tf.nn.softmax(tf.squeeze(g)) # (batch, len)
    #   alpha = tf.expand_dims(alpha, axis=-1)
    #   att_out = tf.reduce_sum(alpha * context, axis=1) # (batch, d)

    #   linear_out = tf.layers.dense(entities, self.embed_dim)
    #   linear_out = tf.nn.relu(linear_out)

    #   entities = tf.concat([linear_out, att_out], axis=1)
    #   entities.set_shape([None, 2*310])

    # entities = tf.layers.dropout(entities, 1-FLAGS.keep_prob, training=self.is_train)
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