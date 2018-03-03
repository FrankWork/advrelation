import os
import numpy as np
import tensorflow as tf
from models.adv import *
from models.attention import *


class BaseModel(object):
  def __init__(self, hparams, ini_word_embed, batched_data, is_train):
    self.is_train = is_train
    self.hparams = hparams

    # embedding initialization
    self.word_embed = tf.get_variable('word_embed', initializer= ini_word_embed,
                      dtype=tf.float32, trainable=self.hparams.tune_word_embed)
    pos_shape = [self.hparams.pos_num, self.hparams.pos_dim]  
    self.pos1_embed = tf.get_variable('pos1_embed', shape=pos_shape)
    self.pos2_embed = tf.get_variable('pos2_embed', shape=pos_shape)
    self.embed_dim = self.hparams.word_embed_size + 2*self.hparams.pos_dim

    self.tensors = dict()

    initializer = tf.keras.initializers.he_normal()
    self.regularizer = tf.contrib.layers.l2_regularizer(self.hparams.l2_scale)

    with tf.variable_scope('model_graph', initializer=initializer):
      self.build_graph(batched_data)
    
    self.set_saver()

  def set_saver(self):
    # shared between train and valid model instance
    self.saver = tf.train.Saver(var_list=None)
    self.save_dir = os.path.join(self.hparams.logdir, self.hparams.save_dir)
    self.save_path = os.path.join(self.save_dir, "model.ckpt")

  def restore(self, session):
    ckpt = tf.train.get_checkpoint_state(self.save_dir)
    self.saver.restore(session, ckpt.model_checkpoint_path)

  def save(self, session, global_step):
    self.saver.save(session, self.save_path, global_step)

  def optimize(self, loss, lrn_rate, max_norm=None, decay_steps=None):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch_norm
    with tf.control_dependencies(update_ops):
      global_step = tf.train.get_or_create_global_step()
      
      if decay_steps is not None:
        lrn_rate = tf.train.exponential_decay(lrn_rate, global_step, 
                                      decay_steps, 0.95, staircase=True)
      
      optimizer = tf.train.AdamOptimizer(lrn_rate)
      gradients, variables = zip(*optimizer.compute_gradients(loss))
      
      if max_norm is not None:
        gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
      train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
      return train_op

  def build_graph(self, batched_data):
    raise NotImplementedError

def conv_block_v2(inputs, kernel_size, num_filters, name, training, 
               batch_norm=False, initializer=None, shortcut=None, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    conv_out = tf.layers.conv1d(inputs, num_filters, kernel_size,
                  strides=1, padding='same',
                  kernel_initializer=initializer,
                  name='conv-%d' % kernel_size,
                  reuse=reuse)
    if batch_norm:
      conv_out = tf.layers.batch_normalization(conv_out, training=training)
    conv_out = tf.nn.relu(conv_out)
    if shortcut is not None:
      conv_out = conv_out + shortcut
    return conv_out

class CNNModel(BaseModel):

  def bottom(self, data):
    (labels, length, ent_pos, sentence, pos1, pos2) = data

    # embedding lookup
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    sentence = tf.layers.dropout(sentence, self.hparams.dropout_rate, training=self.is_train)

    return labels, length, ent_pos, sentence, pos1, pos2
  
  def conv_shallow(self, inputs, name='conv_block'):
    conv_out = conv_block_v2(inputs, self.hparams.kernel_size, 
                  self.hparams.num_filters, name,training=self.is_train, 
                  reuse=tf.AUTO_REUSE)
    max_len = self.hparams.max_len
    pool_out = tf.layers.max_pooling1d(conv_out, max_len, max_len, padding='same')
    pool_out = tf.squeeze(pool_out, axis=1)
    return pool_out

  def compute_logits(self, sentence, length, ent_pos, pos1, pos2, regularizer=None):
    inputs = tf.concat([sentence, pos1, pos2], axis=2)

    entities = self.slice_entity(inputs, ent_pos, length)
    depth = self.hparams.word_embed_size + 2*self.hparams.pos_dim
    scaled_entities = multihead_attention(entities, inputs, None, depth, 
                                  depth, depth, 10, reuse=tf.AUTO_REUSE,
                                  name='ent-mh-att')
    ent_out = tf.nn.relu(scaled_entities)
    ent_out = tf.reduce_max(ent_out, axis=1)
    # ent_out = self.conv_shallow(scaled_entities, 'conv_ent')

    conv_out = self.conv_shallow(inputs)
    # conv_out = self.conv_deep(inputs)

    out = tf.concat([ent_out, conv_out], axis=1)
    if not self.hparams.tune_conv:
      out = tf.stop_gradient(out)

    # out = conv_out
    out = tf.layers.dropout(out, self.hparams.dropout_rate, training=self.is_train)
    logits = tf.layers.dense(out, self.hparams.num_classes, 
                        name='logits-%d' % self.hparams.num_classes,
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
    
    depth = self.hparams.word_embed_size + 2*self.hparams.pos_dim

    entities = slice_batch_n(inputs, [begin1, begin2], [size1, size2])
    entities.set_shape(tf.TensorShape([None, None, depth]))

    return entities

  def compute_xentropy_loss(self, logits, labels):
    # Calculate Mean cross-entropy loss
    with tf.name_scope("loss"):
      one_hot = tf.one_hot(labels, self.hparams.num_classes)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                  labels=one_hot)
      # cross_entropy = tf.reduce_mean(focal_loss(one_hot, logits))

    return tf.reduce_mean(cross_entropy)
  
  def build_graph(self, data):
    labels, length, ent_pos, sentence, pos1, pos2 = self.bottom(data)

    # cross entropy loss
    logits = self.compute_logits(sentence, length, ent_pos, pos1, pos2, regularizer=self.regularizer)
    loss_xent = self.compute_xentropy_loss(logits, labels)

    # # adv loss
    # adv_sentence = adv_example(sentence, loss_xent)
    # adv_logits = self.compute_logits(adv_sentence, length, ent_pos, pos1, pos2)
    # loss_adv = self.compute_xentropy_loss(adv_logits, labels)

    # # vadv loss
    # loss_vadv = virtual_adversarial_loss(logits, sentence, length, ent_pos, pos1, pos2, self.compute_logits)

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
    self.tensors['loss'] = loss_xent + loss_l2 # + loss_adv + loss_vadv
    self.tensors['pred'] = pred

    self.maybe_build_train_op()

  def maybe_build_train_op(self):
    if not self.is_train:
      return

    self.train_ops = dict()
    loss = self.tensors['loss']
    self.train_ops['train_loss'] = self.optimize(loss, self.hparams.learning_rate)

  def train_epoch(self, session, num_batches_per_epoch):
    if not self.is_train:
      return

    moving_loss, moving_acc = [], []
    for batch in range(num_batches_per_epoch):
      train_op = self.train_ops['train_loss']
      fetches = [train_op, self.tensors['loss'], self.tensors['acc']]
      _, loss, acc = session.run(fetches)

      moving_loss.append(loss)
      moving_acc.append(acc)
   
    return np.mean(moving_loss), np.mean(moving_acc)*100
  
  def train_step(self, session):
    if not self.is_train:
      return

    fetches = [self.train_ops['train_loss'], self.tensors['loss'], self.tensors['acc']]
    _, loss, acc = session.run(fetches)

    return loss, acc


  def evaluate(self, session, test_ds_iter, num_batches):
    if self.is_train:
      return

    session.run(test_ds_iter.initializer)

    moving_acc = []
    for batch in range(num_batches):
      acc = session.run(self.tensors['acc'])
      moving_acc.append(acc)
    
    return np.mean(moving_acc)*100
  
  def pred_results(self, session, test_ds_iter, num_batches):
    if self.is_train:
      return

    session.run(test_ds_iter.initializer)

    all_pred = []
    for batch in range(num_batches):
      pred = session.run(self.tensors['pred'])
      all_pred.append(pred)
    all_pred = np.concatenate(all_pred)

    return all_pred


def build_train_valid_model(hparams, ini_word_embed, train_data, test_data):
  with tf.name_scope("Train"):
    with tf.variable_scope('CNNModel', reuse=tf.AUTO_REUSE):
      m_train = CNNModel(hparams, ini_word_embed, train_data, is_train=True)
  with tf.name_scope('Valid'):
    with tf.variable_scope('CNNModel', reuse=True):
      m_valid = CNNModel(hparams, ini_word_embed, test_data, is_train=False)
  return m_train, m_valid
