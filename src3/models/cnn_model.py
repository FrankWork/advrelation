def build_train_valid_model(model_name, word_embed, train_data, test_data):
  with tf.name_scope("Train"):
    with tf.variable_scope(model_name, reuse=None):
      m_train = CNNModel(word_embed, train_data, is_train=True)
      m_train.set_saver(model_name)
      m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope(model_name, reuse=True):
      m_valid = CNNModel(word_embed, test_data, is_train=False)
      m_valid.set_saver(model_name)
  
  return m_train, m_valid