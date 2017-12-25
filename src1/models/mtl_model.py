import tensorflow as tf
from models.base_model import * 

flags = tf.app.flags

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

FLAGS = flags.FLAGS

MAX_LEN = 97
SEM_CLASS_NUM = 19
DB_CLASS_NUM = 14
CLASS_NUM = SEM_CLASS_NUM + DB_CLASS_NUM
TASK_NUM = 2

class MTLModel(BaseModel):
  '''Multi Task Learning'''

  def __init__(self, word_embed, semeval_data, dbpedia_data, is_mtl, is_adv, is_train):
    # input data
    self.is_train = is_train
    self.is_adv = is_adv
    self.is_mtl = is_mtl

    # embedding initialization
    self.word_dim = word_embed.shape[1]
    w_trainable = True if self.word_dim==50 else False
    
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer=word_embed,
                                      dtype=tf.float32,
                                      trainable=w_trainable)
    pos_shape = [FLAGS.pos_num, FLAGS.pos_dim]  
    self.pos1_embed = tf.get_variable('pos1_embed', shape=pos_shape)
    self.pos2_embed = tf.get_variable('pos2_embed', shape=pos_shape)

    self.shared_conv = ConvLayer('conv_shared', FILTER_SIZES)
    self.shared_linear = LinearLayer('linear_shared', TASK_NUM, True)

    self.tensors = []

    with tf.variable_scope('dbpedia_graph'):
      self.build_dbpedia_graph(dbpedia_data)
    with tf.variable_scope('semeval_graph'):
      self.build_semeval_graph(semeval_data)

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
    loss_adv = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=task_label, logits=logits))

    return loss_adv, loss_l2
  
  def diff_loss(self, shared_feat, task_feat):
    '''Orthogonality Constraints from https://github.com/tensorflow/models,
    in directory research/domain_adaptation
    '''
    task_feat -= tf.reduce_mean(task_feat, 0)
    shared_feat -= tf.reduce_mean(shared_feat, 0)

    task_feat = tf.nn.l2_normalize(task_feat, 1)
    shared_feat = tf.nn.l2_normalize(shared_feat, 1)

    correlation_matrix = tf.matmul(
        task_feat, shared_feat, transpose_a=True)

    cost = tf.reduce_mean(tf.square(correlation_matrix)) * 0.01
    cost = tf.where(cost > 0, cost, 0, name='value')

    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
      loss_diff = tf.identity(cost)

    return loss_diff

  def build_semeval_graph(self, data):
    lexical, labels, sentence, pos1, pos2 = data

    # embedding lookup
    lexical = tf.nn.embedding_lookup(self.word_embed, lexical)
    lexical = tf.reshape(lexical, [-1, 6*self.word_dim])

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    pos1 = tf.nn.embedding_lookup(self.pos1_embed, pos1)
    pos2 = tf.nn.embedding_lookup(self.pos2_embed, pos2)

    # cnn model
    if self.is_train:
      sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    sent_pos = tf.concat([sentence, pos1, pos2], axis=2)
    
    conv_layer = ConvLayer('conv_semeval', FILTER_SIZES)
    conv_out = conv_layer(sent_pos)
    conv_out = max_pool(conv_out, MAX_LEN)

    shared_out = self.shared_conv(sentence)
    shared_out = max_pool(shared_out, MAX_LEN)
  
    if self.is_mtl:
      feature = tf.concat([lexical, conv_out, shared_out], axis=1)
    else:
      feature = tf.concat([lexical, conv_out], axis=1)

    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 19 classes
    linear = LinearLayer('linear_semeval', CLASS_NUM, True)
    logits, loss_l2 = linear(feature)

    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                                  labels=tf.one_hot(labels, CLASS_NUM), 
                                  logits=logits)
    loss_ce = tf.reduce_mean(xentropy)

    task_label = tf.one_hot(tf.ones_like(labels), 2)
    loss_adv, loss_adv_l2 = self.adversarial_loss(shared_out, task_label)
    loss_diff = self.diff_loss(shared_out, conv_out)

    if self.is_adv:
      loss = loss_ce + 0.05*loss_adv + FLAGS.l2_coef*(loss_l2+loss_adv_l2) #+ loss_diff
    else:
      loss = loss_ce  + FLAGS.l2_coef*loss_l2

    pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(pred, labels), tf.float32)
    acc = tf.reduce_mean(acc)

    self.tensors.append((acc, loss, pred))

  def build_dbpedia_graph(self, data):
    labels, entity, sentence = data

    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)
    entity = tf.nn.embedding_lookup(self.word_embed, entity)
    entity = tf.reshape(entity, [-1, 3*self.word_dim])

    if self.is_train:
      sentence = tf.nn.dropout(sentence, FLAGS.keep_prob)
    
    conv_layer = ConvLayer('conv_dbpedia', FILTER_SIZES)
    conv_out = conv_layer(sentence)
    conv_out = max_pool(conv_out, MAX_LEN)

    shared_out = self.shared_conv(sentence)
    shared_out = max_pool(shared_out, MAX_LEN)

    if self.is_mtl:
      feature = tf.concat([entity, conv_out, shared_out], axis=1)
    else:
      feature = tf.concat([entity, conv_out], axis=1)

    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 14 classes
    linear = LinearLayer('linear_dbpedia', CLASS_NUM, True)
    logits, loss_l2 = linear(feature)

    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                                  labels=tf.one_hot(labels, CLASS_NUM), 
                                  logits=logits)
    loss_ce = tf.reduce_mean(xentropy)

    task_label = tf.one_hot(tf.zeros_like(labels), 2)
    loss_adv, loss_adv_l2 = self.adversarial_loss(shared_out, task_label)
    loss_diff = self.diff_loss(shared_out, conv_out)

    if self.is_adv:
      loss = loss_ce + 0.05*loss_adv + FLAGS.l2_coef*(loss_l2+loss_adv_l2) + loss_diff
    else:
      loss = loss_ce + FLAGS.l2_coef*loss_l2

    pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(pred, labels), tf.float32)
    acc = tf.reduce_mean(acc)

    self.tensors.append((acc, loss, pred))

  def build_train_op(self):
    if self.is_train:
      self.train_ops = []
      
      _, loss, _ = self.tensors[0] 
      train_op = optimize(loss, 0.0001)
      self.train_ops.append(train_op)

      _, loss, _ = self.tensors[1] 
      train_op = optimize(loss, 0.001)
      self.train_ops.append(train_op)


def build_train_valid_model(model_name, word_embed, 
                            semeval_train, semeval_test, 
                            dbpedia_train, dbpedia_test, is_mtl, is_adv):
  with tf.name_scope("Train"):
    with tf.variable_scope('MTLModel', reuse=None):
      m_train = MTLModel(word_embed, semeval_train, dbpedia_train, is_mtl, is_adv, is_train=True)
      m_train.set_saver(model_name)
      m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope('MTLModel', reuse=True):
      m_valid = MTLModel(word_embed, semeval_test, dbpedia_test, is_mtl, is_adv, is_train=False)
      m_valid.set_saver(model_name)
  return m_train, m_valid