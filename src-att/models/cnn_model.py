import tensorflow as tf
from models.base_model import * 
from models.attention import *
from models.residual import residual_net

import tensorflow.contrib.eager as tfe

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_integer("num_hops", 1, "hop numbers of entity attention")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("dropout_rate", 0.5, "dropout probability")


FLAGS = flags.FLAGS

MAX_LEN = 98
NUM_CLASSES = 19
# NUM_POWER_ITER = 1
# SMALL_CONSTANT = 1e-6
KERNEL_SIZE = 3
NUM_FILTERS = 310

class EmbedLayer(tf.layers.Layer):
  def __init__(self, name, pretrained_embed=None, tune=False, shape=None, **kwargs):
    if pretrained_embed is not None:
      self.embed = tf.get_variable(name, initializer=pretrained_embed,
                                   dtype=tf.float32, trainable=tune)
    else:
      assert shape is not None
      self.embed = tf.get_variable(name, shape=shape)

    super(EmbedLayer, self).__init__(**kwargs)
  
  def call(self, x):
    return tf.nn.embedding_lookup(self.embed, x)

class CNNShallowNetwork(tfe.Network):
  def __init__(self, word_embed, pos1_embed, pos2_embed):
    super(CNNShallowNetwork, self).__init__()
    self.is_train = True

    self.word_embed = self.track_layer(word_embed)
    self.pos1_embed = self.track_layer(pos1_embed)
    self.pos2_embed = self.track_layer(pos2_embed)

    self.input_drop_layer = self.track_layer(
      tf.layers.Dropout(FLAGS.dropout_rate)
    )

    self.conv_layer = self.track_layer(
      tf.layers.Conv1D(NUM_FILTERS, KERNEL_SIZE, strides=1, padding='same', 
                       activation=tf.nn.relu,
                       kernel_initializer=tf.keras.initializers.he_normal())
      )
    self.pool_layer = self.track_layer(
      tf.layers.MaxPooling1D(MAX_LEN, MAX_LEN, padding='same')
    )
    self.output_drop_layer = self.track_layer(
      tf.layers.Dropout(FLAGS.dropout_rate)
    )
    self.dense_layer = self.track_layer(
      tf.layers.Dense(NUM_CLASSES)
    )

  def call(self, inputs):
    length, ent_pos, sentence, position1, position2 = inputs

    sentence = self.word_embed(sentence)
    pos1 = self.pos1_embed(position1)
    pos2 = self.pos2_embed(position2)

    if self.is_train:
      sentence = self.input_drop_layer(sentence)
    x = tf.concat([sentence, pos1, pos2], axis=2)

    conv_out = self.conv_layer(x)
    pool_out = self.pool_layer(conv_out)
    pool_out = tf.squeeze(pool_out, axis=1)
    if self.is_train:
      pool_out = self.output_drop_layer(pool_out)
    return self.dense_layer(pool_out)

class CNNModel(BaseModel):

  def __init__(self, word_embed, is_adv):
    self.is_adv = is_adv

    # embedding initialization
    word_embed = EmbedLayer('word_embed', pretrained_embed=word_embed)
    pos_shape = [FLAGS.pos_num, FLAGS.pos_dim]  
    pos1_embed = EmbedLayer('pos1_embed', shape=pos_shape)
    pos2_embed = EmbedLayer('pos2_embed', shape=pos_shape)

    self.network = CNNShallowNetwork(word_embed, pos1_embed, pos2_embed)

    super(CNNModel, self).__init__()

  def set_train_mode(self):
    self.network.is_train = True
  
  def set_test_mode(self):
    self.network.is_train = False

  def forward(self, data):
    label, length, ent_pos, sentence, position1, position2 = data
    inputs = (length, ent_pos, sentence, position1, position2)
    logits = self.network(inputs)

    return label, logits
  
  def loss(self, data):
    labels, logits = self.forward(data)
    self.tensors['acc'] = self.accuracy(logits, labels)

    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      one_hot = tf.one_hot(labels, NUM_CLASSES)
      # one_hot = label_smoothing(one_hot)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot,
                                logits=logits)

      l2_loss = tf.constant(0, dtype=tf.float32)
      for tensor in self.network.dense_layer.variables:
        l2_loss += FLAGS.l2_coef * tf.nn.l2_loss(tensor)

      loss = tf.reduce_mean(cross_entropy) + l2_loss

    return loss

  def extract_ent_context(self, inputs, ent_pos):
    '''
    Args:
      inputs: [batch, len, feat]
      ent_pos : [batch, 4]
    
    Returns:
      ent1:   [batch, max_ent_len, feat]
      ent2:   [batch, max_ent_len, feat]
      context: [batch, max_cont_len, feat]
    '''
    # inputs length range
    n_len = inputs.shape.as_list()[1]
    len_range = tf.range(n_len)
    len_range = tf.expand_dims(len_range, axis=0)# (1, len)

    def entity_mask(len_range, start, end):
      '''
      Args
        len_range: [1, len]
        start: [] => [batch, 1]
        end:   [] => [batch, 1]

      Returns
        mask: [batch, len, 1]
      '''
      # mask entity
      m0 = tf.greater_equal(len_range, tf.expand_dims(start, axis=-1))
      m1 = tf.less_equal(len_range, tf.expand_dims(end, axis=-1))
      mask = tf.logical_and(m0, m1)
      return tf.expand_dims(mask, -1)
    
    mask1 = entity_mask(len_range, ent_pos[:, 0], ent_pos[:, 1])
    mask2 = entity_mask(len_range, ent_pos[:, 2], ent_pos[:, 3])

    mask = tf.logical_or(mask1, mask2)
    mask = tf.logical_not(mask)

    tensor_list = tf.unstack(inputs, axis=0)

    def extract(mask, tensor_list):
      '''
      Extract entity or context based on mask

      Args
        mask: [batch, len, 1]
        tensor_list: a list of tensors, each has shape [1, len, feat]
      Returns
        a tensor of shape [batch, max_len, feat]
      '''
      n_feat = tensor_list[0].shape.as_list()[-1]
      mask = tf.tile(mask, [1, 1, n_feat])

      mask_list = tf.unstack(mask, axis=0)
      tensor_list = []
      for t, mask in zip(tensor_list, mask_list):
        t = tf.boolean_mask(t, mask)
        t = tf.reshape(t, [-1, n_feat])
        tensor_list.append(t)

      # pad extracted tensor, and stack as a tensor
      max_len = tf.reduce_max([tensor.shape[0] for tensor in tensor_list])
      tensor_list = [tf.pad(t, [[0, max_len-t.shape[0]], [0, 0]]) for t in tensor_list]
      tensor = tf.stack(tensor_list)
      return tensor
    
    ent1 = extract(mask1, tensor_list)
    ent2 = extract(mask2, tensor_list)
    context = extract(mask, tensor_list)
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

  def entity_attention(self, context, ent1, ent2, num_hops):
    cont1 = context
    cont2 = context
    for i in range(num_hops):
      cont1 = multihead_attention(cont1, ent1, num_heads=10, 
                                  dropout_rate=FLAGS.dropout_rate,
                                  is_training=self.is_train, scope='att1%d'%i)
      cont2 = multihead_attention(cont2, ent2, num_heads=10, 
                                  dropout_rate=FLAGS.dropout_rate,
                                  is_training=self.is_train, scope='att2%d'%i)
      # cont1 = feedforward(cont1, num_units=[620, 310], scope='ffd1%d'%i)
      # cont2 = feedforward(cont2, num_units=[620, 310], scope='ffd2%d'%i)
      ent1 = cont1
      ent2 = cont2

    ent1 = tf.reduce_mean(ent1, axis=1) # (batch, embed)
    ent2 = tf.reduce_mean(ent2, axis=1)
    entities = tf.concat([ent1, ent2], axis=-1)
    # entities = tf.squeeze(entities, axis=1)
    
    return entities

  def prediction(self, logits):
    return tf.argmax(logits, axis=1)

  def accuracy(self, logits, labels):
    pred = self.prediction(logits)
    acc = tf.cast(tf.equal(pred, labels), tf.float32)
    return tf.reduce_mean(acc)
