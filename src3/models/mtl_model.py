import tensorflow as tf
from models.base_model import * 

FLAGS = tf.app.flags.FLAGS

TASK_NUM=14

class MTLModel(BaseModel):

  def __init__(self, word_embed, all_data, adv, is_train):
    # input data
    # self.all_data = all_data
    self.is_train = is_train
    self.adv = adv

    # embedding initialization
    self.word_dim = word_embed.shape[1]
    w_trainable = True if self.word_dim==50 else False
    
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer=word_embed,
                                      dtype=tf.float32,
                                      trainable=w_trainable)
    
    self.shared_conv = ConvLayer('conv_shared', FILTER_SIZES)
    self.shared_linear = LinearLayer('linear_shared', TASK_NUM, True)

    self.tensors = []

    for task_name, data in all_data:
      with tf.name_scope(task_name):
        self.build_task_graph(data)

  def adversarial_loss(self, feature, task_label):
    '''make the task classifier cannot reliably predict the task based on 
    the shared feature
    '''
    # input = tf.stop_gradient(input)
    feature = flip_gradient(feature)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to TASK_NUM classes
    logits, loss_l2 = self.shared_linear(feature)

    label = tf.one_hot(task_label, TASK_NUM)
    loss_adv = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    # Orthogonality Constraints
    # loss_diff += tf.reduce_sum(
    #                   tf.square(
    #                     tf.matmul(cnn_out, shared, transpose_a=True)
    #                   ))

    return loss_adv, loss_l2

  def build_task_graph(self, data):
    task_label, labels, sentence = data
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    if self.is_train:
      sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    
    conv_layer = ConvLayer('conv_task', FILTER_SIZES)
    conv_out = conv_layer(sentence)
    conv_out = max_pool(conv_out, 500)

    shared_out = self.shared_conv(sentence)
    shared_out = max_pool(shared_out, 500)

    if self.adv:
      feature = tf.concat([conv_out, shared_out], axis=1)
    else:
      feature = conv_out

    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 2 classes
    linear = LinearLayer('linear', 2, True)
    logits, loss_l2 = linear(feature)
    
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                          labels=tf.one_hot(labels, 2), 
                          logits=logits)
    loss_ce = tf.reduce_mean(xentropy)

    loss_adv, loss_adv_l2 = self.adversarial_loss(shared_out, task_label)

    if self.adv:
      loss = loss_ce + 0.05*loss_adv + FLAGS.l2_coef*(loss_l2+loss_adv_l2)
    else:
      loss = loss_ce  + FLAGS.l2_coef*loss_l2
    
    pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(pred, labels), tf.float32)
    acc = tf.reduce_mean(acc)

    self.tensors.append((acc, loss))
    # self.tensors.append((acc, loss, pred))
    
  def build_train_op(self):
    if self.is_train:
      self.train_ops = []
      for _, loss in self.tensors:
        train_op = optimize(loss)
        self.train_ops.append(train_op)

def build_train_valid_model(model_name, word_embed, all_train, all_test, adv):
  with tf.name_scope("Train"):
    with tf.variable_scope(model_name, reuse=None):
      m_train = MTLModel(word_embed, all_train, adv, is_train=True)
      m_train.set_saver(model_name)
      m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope(model_name, reuse=True):
      m_valid = MTLModel(word_embed, all_test, adv, is_train=False)
      m_valid.set_saver(model_name)
  
  return m_train, m_valid