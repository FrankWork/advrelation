import tensorflow as tf
from models.base_model import * 
from models.attention import *
from models.residual import residual_net

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_integer("num_hops", 1, "hop numbers of entity attention")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("dropout_rate", 0.3, "dropout probability")
flags.DEFINE_float("lrn_rate", 0.001, "learning rate")

FLAGS = flags.FLAGS

MAX_LEN = 98
NUM_CLASSES = 19
# NUM_POWER_ITER = 1
# SMALL_CONSTANT = 1e-6
KERNEL_SIZE = 3
NUM_FILTERS = 310

class CNNModel(BaseModel):

  def __init__(self, word_embed, vocab_freq, semeval_data, is_adv, is_train):
    # input data
    self.is_train = is_train
    self.is_adv = is_adv

    self.he_normal = tf.keras.initializers.he_normal()
    self.regularizer = tf.contrib.layers.l2_regularizer(FLAGS.l2_coef)

    # embedding initialization
    self.vocab_size, self.word_dim = word_embed.shape
    self.word_dim = 100
    w_trainable = True #if self.word_dim==50 else False
    
    # initializer=word_embed
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

  def bottom(self, data):
    (label, length, sentence, position1, position2) = data

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, position1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, position2)

    # sentence = tf.concat([sentence, pos1, pos2], axis=-1)

    return label, sentence

  def body(self, X):
    num_filters1 = 150
    num_filters2 = 100

    filter1_size=2
    filter2_size=5
    seq_len = MAX_LEN

    # LSTM layer
    with tf.variable_scope('lstm'):
      lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters1, state_is_tuple=True)
      lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters1, state_is_tuple=True)
      
      _X = tf.unstack(X, num=seq_len, axis=1)
      outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs = _X, dtype=tf.float32)
      outputs = tf.stack(outputs, axis=1)
      
      h1_rnn = tf.expand_dims(outputs, -1)            

       ## Max pooling
      h1_pool = tf.nn.max_pool(h1_rnn,ksize=[1, filter1_size, 1, 1],strides=[1, 1, 1, 1], padding='VALID')         
      
    
    
    # CNN+Maxpooling Layer
  
    with tf.variable_scope('cnn'):
      filter_shape = [filter2_size, 2*num_filters1, 1, num_filters2]
      W_cnn = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")    #Convolution parameter
      b_cnn = tf.Variable(tf.constant(0.1, shape=[num_filters2]), name="b")      #Convolution bias parameter
      conv = tf.nn.conv2d(h1_pool, 
            W_cnn, 
            strides=[1, 1, 1, 1], 
            padding="VALID", 
            name="conv") 
      h1_cnn = tf.nn.relu(tf.nn.bias_add(conv, b_cnn))            
      
      ##Maxpooling
      h2_pool=tf.nn.max_pool(h1_cnn, 
            ksize=[1,seq_len-(filter1_size-1)-(filter2_size-1),1,1],
            strides=[1, 1, 1, 1],
            padding="VALID")
      h2_cnn = tf.squeeze(h2_pool, axis=[1,2])

      
    
      ##Dropout
    h_flat = tf.reshape(h2_pool,[-1,num_filters2])
    # h_flat = tf.reshape(h2_cnn,[-1,(seq_len-3*(filter_size-1))*2*num_filters])
    # h_drop = tf.nn.dropout(h_flat,self.dropout_keep_prob)
    body_out = h_flat
    
    body_out = tf.layers.dropout(body_out, FLAGS.dropout_rate, training=self.is_train)
    return body_out

  def body_conv(self, inputs):
    conv_out = conv_block_v2(inputs, KERNEL_SIZE, NUM_FILTERS,
                            'conv_block1',training=self.is_train, 
                          initializer=self.he_normal, batch_norm=False)

    pool_out = tf.layers.max_pooling1d(conv_out, MAX_LEN, MAX_LEN, padding='same')
    pool_out = tf.squeeze(pool_out, axis=1)
    
    # h1_cnn = conv_out
    # ## Attentive pooling
    # W_a1 = tf.get_variable("W_a1", shape=[NUM_FILTERS, NUM_FILTERS])        # 100x100
    # tmp1 = tf.matmul(tf.reshape(h1_cnn, shape=[-1, NUM_FILTERS]), W_a1, name="Wy")   # NMx100
    # h2_cnn = tf.reshape(tmp1, shape=[-1, MAX_LEN, NUM_FILTERS])         #NxMx100

    # M = tf.tanh(h2_cnn)                  # NxMx100
    # W_a2 = tf.get_variable("W_a2", shape=[NUM_FILTERS, 1])         # 100 x 1
    # tmp3 = tf.matmul(tf.reshape(M, shape=[-1, NUM_FILTERS]), W_a2)      # NMx1
    # alpha = tf.nn.softmax(tf.reshape(tmp3, shape=[-1, MAX_LEN], name="att"))  # NxM  
    # self.ret_alpha = alpha

    # alpha = tf.expand_dims(alpha, 1)             # Nx1xM
    # h2_pool = tf.matmul(alpha, tf.reshape(h1_cnn, shape=[-1, MAX_LEN, NUM_FILTERS]), name="r")
    # h2_pool = tf.squeeze(h2_pool, axis=1)

    body_out = tf.layers.dropout(pool_out, FLAGS.dropout_rate, training=self.is_train)
    return body_out

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
    labels, sentence = self.bottom(data)
    body_out = self.body(sentence)
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