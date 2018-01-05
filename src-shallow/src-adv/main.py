import os
import time
import sys
import tensorflow as tf
import numpy as np

# from inputs import semeval_v2 as semeval
from inputs import util
from inputs import semeval
from models import cnn_model

# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_integer("word_dim", 300, "word embedding size")
flags.DEFINE_integer("num_epochs", 200, "number of epochs")
flags.DEFINE_integer("batch_size", 100, "batch size")

flags.DEFINE_boolean('build_data', False, 'set True to generate data')
flags.DEFINE_boolean('is_adv', False, 'set True to use adv training')
flags.DEFINE_boolean('is_test', False, 'set True to test')

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def build_data():
  '''load raw data, build vocab, build TFRecord data, trim embeddings
  '''
  def _build_data(semeval_train, semeval_test):
    vocab2id = semeval.load_vocab2id()

    print('convert semeval data to TFRecord')
    semeval.write_as_tfrecord(semeval_train, semeval_test, vocab2id)

  def _trim_embed():
    print('trimming pretrained embeddings')
    util.trim_embeddings(50)
    util.trim_embeddings(300)

  print('load raw data')
  semeval_train, semeval_test = semeval.load_raw_data(verbose=True)

  print('build vocab')
  semeval.build_vocab(semeval_train + semeval_test)

  _build_data(semeval_train, semeval_test)

  _trim_embed()

def train_semeval(sess, m_train, m_valid, semeval_test_iter):
  best_acc, best_epoch = 0., 0
  start_time = time.time()
  orig_begin_time = start_time

  for epoch in range(FLAGS.num_epochs):
    sess.run([semeval_test_iter.initializer])

    # train SemEval
    sem_loss, sem_acc = 0., 0.
    for batch in range(80):
      fetches = [m_train.train_op, m_train.tensors['loss'], m_train.tensors['acc']]
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

def test(sess, m_valid, semeval_test_iter):
  m_valid.restore(sess)
  sess.run([semeval_test_iter.initializer])
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
  semeval.write_results(pred_all)

def main(_):
  if FLAGS.build_data:
    build_data()
    exit()

  _, vocab_freq = semeval.load_vocab_and_freq()
  word_embed = util.load_embedding(word_dim=FLAGS.word_dim)
  with tf.Graph().as_default():
    semeval_train_iter, semeval_test_iter = semeval.read_tfrecord(
                                          FLAGS.num_epochs, FLAGS.batch_size)
    model_name = 'cnn-%d-%d' % (FLAGS.word_dim, FLAGS.num_epochs)
    semeval_train = semeval_train_iter.get_next()
    semeval_test = semeval_test_iter.get_next()
    m_train, m_valid = cnn_model.build_train_valid_model(
                          model_name, word_embed, vocab_freq,
                          semeval_train, semeval_test, 
                          FLAGS.is_adv, FLAGS.is_test)
    
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      print('='*80)

      if FLAGS.is_test:
        test(sess, m_valid, semeval_test_iter)
      else:
        train_semeval(sess, m_train, m_valid, semeval_test_iter)

if __name__ == '__main__':
  tf.app.run()
