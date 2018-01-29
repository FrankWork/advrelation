import tensorflow as tf
from models.base_model import * 
# from models.adv import *
# from models.attention import *


MAX_LEN = 97
NUM_CLASSES = 19
KERNEL_SIZE = 3
NUM_FILTERS = 310

class RNNModel(BaseModel):

  def __init__(self, hparams, ini_word_embed, semeval_data, is_train):
    self.is_train = is_train
    self.hparams = hparams

    # embedding initialization
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer= ini_word_embed,
                                      dtype=tf.float32,
                                      trainable=False)
    
    self.tensors = dict()

    initializer = tf.keras.initializers.he_normal()
    self.regularizer = tf.contrib.layers.l2_regularizer(self.hparams.l2_scale)

    with tf.variable_scope('tag_graph', initializer=self.initializer):
      self.build_semeval_graph(semeval_data)

  def embed_layer(self, sentence):
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    sentence = tf.layers.dropout(sentence, self.hparams.dropout_rate, 
                                 training=self.is_train)

    return sentence
  
  def compute_logits(self, inputs, lengths):
    with tf.variable_scope("bi-lstm-encoder"):
      cell_fw = tf.contrib.rnn.LSTMCell(self.hparams.hidden_size, 
                                        use_peepholes=True, name='en_cell_fw')
      cell_bw = tf.contrib.rnn.LSTMCell(self.hparams.hidden_size, , 
                                        use_peepholes=True, name='en_cell_bw')
      bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(
                          cell_fw, cell_bw, inputs, 
                          sequence_length=lengths, dtype=tf.float32)
      en_output = tf.concat(bi_output, axis=-1)
      en_output = tf.layers.dropout(en_output, self.hparams.dropout_rate, 
                                 training=self.is_train)

      state_fw, state_bw = bi_state
      en_state =  tf.contrib.rnn.LSTMStateTuple(
            tf.concat([state_fw[0], state_bw[0]], axis=-1), 
            tf.concat([state_fw[1], state_bw[1]], axis=-1))

    with tf.variable_scope("lstm-decoder"):
      cell_de = tf.contrib.rnn.LSTMCell(2*self.hparams.hidden_size, name='de_cell')
      helper = tf.contrib.seq2seq.TrainingHelper(en_output, lengths)
      
      proj_layer = tf.layers.Dense(self.hparams.num_tags, use_bias=False)
      decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell_de, helper, en_state,
                    output_layer=proj_layer)
      # Dynamic decoding
      outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
      logits = outputs.rnn_output

    return logits
  
  def compute_xentropy_loss(self, logits, labels):
    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

    return tf.reduce_mean(cross_entropy)
  
  def build_semeval_graph(self, data):
    (sentence, tags, length) = data
    sentence = self.embed_layer(sentence)

    # cross entropy loss
    logits = self.compute_logits(sentence, length)
    loss_xent = self.compute_xentropy_loss(logits, tags)

    # # # adv loss
    # adv_sentence = adv_example(sentence, loss_xent)
    # adv_logits = self.compute_logits(sentence, length, ent_pos, pos1, pos2)
    # loss_adv = self.compute_xentropy_loss(adv_logits, labels)

    # l2 loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_l2 = sum(regularization_losses)
    
    # Accuracy
    with tf.name_scope("accuracy"):
      pred = tf.argmax(logits, axis=1)
      acc = tf.cast(tf.equal(pred, labels), tf.float32)
      acc = tf.reduce_mean(acc)

    self.tensors['acc'] = acc
    self.tensors['loss'] = loss_xent + loss_adv + loss_l2 #+ loss_vadv
    self.tensors['pred'] = pred

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
    with tf.variable_scope('RNNModel', reuse=None):
      m_train = RNNModel(word_embed, train_data, unsup_data, is_adv, is_train=True)
      m_train.set_saver(model_name)
      if not is_test:
        m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope('RNNModel', reuse=True):
      m_valid = RNNModel(word_embed, test_data, unsup_data, is_adv, is_train=False)
      m_valid.set_saver(model_name)
  return m_train, m_valid
