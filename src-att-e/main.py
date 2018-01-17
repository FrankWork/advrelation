import os
import time
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

from inputs import  dataset, semeval_v2
from models import cnn_model

# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_integer("word_dim", 300, "word embedding size")
flags.DEFINE_integer("num_epochs", 50, "number of epochs")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_float("lrn_rate", 0.001, "learning rate")
flags.DEFINE_boolean('is_adv', False, 'set True to use adv training')
flags.DEFINE_boolean('is_test', False, 'set True to test')

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def test(m_valid, test_iter):
  m_valid.restore(sess)
  sess.run([test_iter.initializer])
  acc_tenser = m_valid.tensors['acc']
  pred_tensor = m_valid.tensors['pred']

  acc_all = 0.
  pred_all = []
  for batch in range(28):
    acc, pred = sess.run([acc_tenser, pred_tensor])
    acc_all += acc
    pred_all.append(pred)
  acc_all /= 28

  print('acc: %.4f' % acc_all)
  # print(type(pred_all[0]))
  # print(pred_all[0].shape)
  pred_all = np.concatenate(pred_all)
  # print(type(pred_all))
  # print(pred_all.shape)
  # exit()
  semeval_v2.write_results(pred_all)

def main(_):
  vocab_mgr = dataset.VocabMgr()
  word_embed = vocab_mgr.load_embedding()
  semeval_record = semeval_v2.SemEvalCleanedRecordData(None)

  # load dataset
  train_data = semeval_record.train_data(FLAGS.num_epochs, FLAGS.batch_size)
  test_data = semeval_record.test_data(1, FLAGS.batch_size)
   
  # model_name = 'cnn-%d-%d' % (FLAGS.word_dim, FLAGS.num_epochs)
  model = cnn_model.CNNModel(word_embed, FLAGS.is_adv)
 
  # for tensor in tf.trainable_variables():
  #   tf.logging.info(tensor.op.name)
  
  model.train_and_eval(FLAGS.num_epochs, 80, FLAGS.lrn_rate, train_data, test_data)

if __name__ == '__main__':
  tf.app.run()
