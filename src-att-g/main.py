import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import  dataset, semeval_v2
from models import cnn_model

# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_integer("word_dim", 300, "word embedding size")
flags.DEFINE_integer("num_epochs", 50, "number of epochs")
flags.DEFINE_integer("batch_size", 100, "batch size")

flags.DEFINE_boolean('is_adv', False, 'set True to use adv training')
flags.DEFINE_boolean('is_test', False, 'set True to test')

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def train_semeval(sess, m_train, m_valid, test_iter):
  best_acc, best_epoch = 0., 0
  start_time = time.time()
  orig_begin_time = start_time
 
  for epoch in range(FLAGS.num_epochs):
    sess.run([test_iter.initializer])

    # train SemEval
    sem_loss, sem_acc = 0., 0.
    for batch in range(80):
      train_op = m_train.train_op
      fetches = [train_op, m_train.tensors['loss'], m_train.tensors['acc']]
      _, loss, acc = sess.run(fetches)
      sem_loss += loss
      sem_acc += acc

    sem_loss /= 80
    sem_acc /= 80

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    sem_valid_acc = 0.
    for batch in range(28):
      acc = sess.run(m_valid.tensors['acc'])
      sem_valid_acc += acc
    sem_valid_acc /= 28

    if best_acc < sem_valid_acc:
      best_acc = sem_valid_acc
      best_epoch = epoch
      m_train.save(sess, epoch)
    
    print("Epoch %d sem %.2f %.2f %.4f time %.2f" % 
             (epoch, sem_loss, sem_acc, sem_valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_epoch, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(sess, m_valid, test_iter):
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

  with tf.Graph().as_default():
    train_iter = semeval_record.train_data(FLAGS.num_epochs, FLAGS.batch_size)
    test_iter = semeval_record.test_data(1, FLAGS.batch_size)
                                          
    model_name = 'cnn-%d-%d' % (FLAGS.word_dim, FLAGS.num_epochs)
    train_data = train_iter.get_next()
    test_data = test_iter.get_next()
    m_train, m_valid = cnn_model.build_train_valid_model(
                          model_name, word_embed,
                          train_data, test_data,
                          FLAGS.is_adv, FLAGS.is_test)

    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
      sess.run(init_op)
      print('='*80)
      for tensor in tf.trainable_variables():
        tf.logging.info(tensor.op.name)
      
      # for tensor in sess.run(m_train.cache):
      #   print(tensor.shape)
      # exit()
    
      if FLAGS.is_test:
        test(sess, m_valid, test_iter)
      else:
        train_semeval(sess, m_train, m_valid, test_iter)

if __name__ == '__main__':
  tf.app.run()
