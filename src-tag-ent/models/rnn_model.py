import os
import numpy as np
import tensorflow as tf
# from models.adv import *
# from models.attention import *
from models.decode import *
from models.focal_loss import focal_loss

class BaseModel(object):
  def __init__(self, config, ini_word_embed, batched_data, is_train):
    self.is_train = is_train
    self.config = config
    self.hparams = config.hparams

    # embedding initialization
    self.word_embed = tf.get_variable('word_embed', initializer= ini_word_embed,
                      dtype=tf.float32, trainable=self.hparams.tune_word_embed)
    
    self.tensors = dict()

    initializer = tf.keras.initializers.he_normal()
    self.regularizer = tf.contrib.layers.l2_regularizer(self.hparams.l2_scale)

    with tf.variable_scope('model_graph', initializer=initializer):
      self.build_graph(batched_data)
    
    self.set_saver()

  def set_saver(self):
    # shared between train and valid model instance
    self.saver = tf.train.Saver(var_list=None)
    self.save_dir = os.path.join(self.config.logdir, self.config.save_dir)
    self.save_path = os.path.join(self.save_dir, "model.ckpt")

  def restore(self, session):
    ckpt = tf.train.get_checkpoint_state(self.save_dir)
    self.saver.restore(session, ckpt.model_checkpoint_path)

  def save(self, session, global_step):
    self.saver.save(session, self.save_path, global_step)

  def optimize(self, loss, lrn_rate, max_norm=None, decay_steps=None):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch_norm
    with tf.control_dependencies(update_ops):
      global_step = tf.Variable(0, name="global_step", trainable=False)
      
      if decay_steps is not None:
        lrn_rate = tf.train.exponential_decay(lrn_rate, global_step, 
                                      decay_steps, 0.95, staircase=True)
      
      optimizer = tf.train.AdamOptimizer(lrn_rate)
      gradients, variables = zip(*optimizer.compute_gradients(loss))
      
      if max_norm is not None:
        gradients, norm = tf.clip_by_global_norm(gradients, max_norm)
        self.tensors['norm'] = norm
      train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
      return train_op

class RNNModel(BaseModel):

  def embed_layer(self, sentence):
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    sentence = tf.layers.dropout(sentence, self.hparams.dropout_rate, 
                                 training=self.is_train)

    return sentence
  
  def compute_logits(self, inputs, lengths):
    inputs = tf.transpose(inputs, [1, 0, 2]) # time_major, (time, batch, dim)

    with tf.variable_scope("bi-lstm-encoder"):
      cell_fw = tf.contrib.rnn.LSTMCell(self.hparams.hidden_size, 
                                        use_peepholes=True, name='en_cell_fw')
      cell_bw = tf.contrib.rnn.LSTMCell(self.hparams.hidden_size, 
                                        use_peepholes=True, name='en_cell_bw')
      bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(
                          cell_fw, cell_bw, inputs, sequence_length=lengths, 
                          dtype=tf.float32, time_major=True)
      en_output = tf.concat(bi_output, axis=-1)
      en_output = tf.layers.dropout(en_output, self.hparams.dropout_rate, 
                                 training=self.is_train)

      state_fw, state_bw = bi_state
      # c = tf.concat([state_fw.c, state_bw.c], axis=-1)
      # h = tf.concat([state_fw.h, state_bw.h], axis=-1)
      c = state_fw.c + state_bw.c
      h = state_fw.h + state_bw.h
      en_state = TagLSTMStateTuple(c, h, tf.zeros_like(c))

    with tf.variable_scope("lstm-decoder"):
      decode_outputs = decode(en_output, en_state, lengths, self.hparams.hidden_size)
    
    with tf.variable_scope("output"):
      logits_tag = tf.layers.dense(decode_outputs, self.hparams.num_tags)
      r = self.attention(tf.transpose(en_output, [1, 0, 2]), 'att')
      logits_rel = tf.layers.dense(r, self.hparams.num_classes)

    return logits_tag, logits_rel
  
  def attention(self, inputs, name, reuse=None):
    H = inputs
    hidden_size = inputs.shape.as_list()[-1]
    with tf.variable_scope(name, reuse=reuse):
        M = tf.nn.tanh(H) # b,n,d
        w = tf.get_variable('w-att',[1, hidden_size])
        batch_size = tf.shape(H)[0]
        alpha = tf.matmul(tf.tile(tf.expand_dims(w, 0), [batch_size, 1, 1]),
                        M, transpose_b=True)
        alpha = tf.nn.softmax(alpha) # b,1,n
        r = tf.matmul(alpha, H) # b, 1, d
        return tf.squeeze(r, axis=1)

  def compute_xentropy_loss(self, logits_tag, logits_rel, labels, tags, lengths):
    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      # onehot_labels = tf.one_hot(labels, self.hparams.num_tags)

      entropy_tags = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits_tag, labels=tags)
      weights = tf.sequence_mask(lengths, dtype=tf.float32)
      entropy_tags = entropy_tags * weights

      entropy_label = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits_rel, labels=labels)

    return tf.reduce_mean(entropy_label) #tf.reduce_mean(entropy_tags) + 
  
  def build_graph(self, data):
    (labels, lengths, sentence, tags) = data
    sentence = self.embed_layer(sentence)
   
    # cross entropy loss
    logits_tag, logits_rel = self.compute_logits(sentence, lengths)
    loss_xent = self.compute_xentropy_loss(logits_tag, logits_rel, labels, tags, lengths)

    # l2 loss
    # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss_l2 = sum(regularization_losses)
    
    # prediction
    with tf.name_scope("prediction"):
      pred_tags = tf.argmax(logits_tag, axis=-1)
      pred_rel = tf.argmax(logits_rel, axis=-1)
      acc = tf.cast(tf.equal(pred_rel, labels), tf.float32)
      acc = tf.reduce_mean(acc)

    self.tensors['loss'] = loss_xent #+ loss_adv + loss_l2
    self.tensors['pred_tags'] = pred_tags
    self.tensors['pred_rel'] = pred_rel
    self.tensors['rel_acc'] = acc
    self.tensors['lengths'] = lengths
    self.tensors['labels'] = labels
    self.tensors['tags'] = tags

    self.maybe_build_train_op()

  def train_epoch(self, session, num_batches_per_epoch):
    if not self.is_train:
      return

    moving_loss, moving_tag_acc, moving_rel_acc = [], [], []
    for batch in range(num_batches_per_epoch):
      train_op = self.train_ops['train_loss']
      fetches = [train_op, self.tensors['lengths'], self.tensors['tags'], 
          self.tensors['pred_tags'], self.tensors['rel_acc'],
          self.tensors['loss']
                ]
      _, lengths, tags, pred_tags, rel_acc, loss = session.run(fetches)
      
      moving_loss.append(loss)
      for lab, lab_pred, length in zip(tags, pred_tags, lengths):
        lab      = lab[:length]
        lab_pred = lab_pred[:length]
        acc = np.mean(np.equal(lab, lab_pred))
        moving_tag_acc.append(acc)
      
      moving_rel_acc.append(rel_acc)
    
    return np.mean(moving_loss), np.mean(moving_tag_acc)*100, np.mean(moving_rel_acc)*100

  def evaluate(self, session, test_ds_iter, num_batches, vocab_tags, return_pred=False):
    if self.is_train:
      return

    session.run(test_ds_iter.initializer)

    tags_acc = []
    rel_accs = []
    pred_result = []
    tags_result = []
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for batch in range(num_batches):
      fetches = [self.tensors['pred_tags'], self.tensors['lengths'], 
                 self.tensors['tags'], self.tensors['rel_acc']]
      preds, lengths, tags, rel_acc = session.run(fetches)
      for lab, lab_pred, length in zip(tags, preds, lengths):
        lab      = lab[:length]
        lab_pred = lab_pred[:length]
        acc = np.mean(np.equal(lab, lab_pred))
        tags_acc.append(acc)
        pred_result.append(lab_pred)
        tags_result.append(lab)
        rel_accs.append(rel_acc)

        lab_chunks      = set(get_chunks(lab, vocab_tags))
        lab_pred_chunks = set(get_chunks(lab_pred, vocab_tags))

        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds   += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
    tags_acc = np.mean(tags_acc)
    rel_accs = np.mean(rel_accs)

    if return_pred:
      return pred_result, tags_result

    return 100*tags_acc, 100*f1, 100*rel_accs

  def maybe_build_train_op(self):
    if not self.is_train:
      return

    self.train_ops = dict()
    loss = self.tensors['loss']
    self.train_ops['train_loss'] = self.optimize(loss, self.hparams.learning_rate)
      
def get_chunks(seq, tags, default_tag='O'):
  """Given a sequence of tags, group entities and their position

  Args:
      seq: [4, 4, 0, 0, ...] sequence of labels
      tags: dict["O"] = 4

  Returns:
      list of (chunk_type, chunk_start, chunk_end)

  Example:
      seq = [4, 5, 0, 3]
      tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
      result = [("PER", 0, 2), ("LOC", 3, 4)]

  """
  default = tags[default_tag]
  idx_to_tag = {idx: tag for tag, idx in tags.items()}
  chunks = []
  chunk_type, chunk_start = None, None
  for i, tok in enumerate(seq):
    # End of a chunk 1
    if tok == default and chunk_type is not None:
      # Add a chunk.
      chunk = (chunk_type, chunk_start, i)
      chunks.append(chunk)
      chunk_type, chunk_start = None, None

    # End of a chunk + start of a chunk!
    elif tok != default:
      tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
      if chunk_type is None:
        chunk_type, chunk_start = tok_chunk_type, i
      elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
        chunk = (chunk_type, chunk_start, i)
        chunks.append(chunk)
        chunk_type, chunk_start = tok_chunk_type, i
    else:
      pass

  # end condition
  if chunk_type is not None:
    chunk = (chunk_type, chunk_start, len(seq))
    chunks.append(chunk)

  return chunks

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[1]
    return tag_class, tag_type

def build_train_valid_model(config, ini_word_embed, train_data, test_data):
  with tf.name_scope("Train"):
    with tf.variable_scope('RNNModel', reuse=None):
      m_train = RNNModel(config, ini_word_embed, train_data, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('RNNModel', reuse=True):
      m_valid = RNNModel(config, ini_word_embed, test_data, is_train=False)
  return m_train, m_valid
