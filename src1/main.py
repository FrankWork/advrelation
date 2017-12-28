import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import util
from inputs import dbpedia
from inputs import semeval
from models import mtl_model

# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_integer("word_dim", 300, "word embedding size")
flags.DEFINE_integer("num_epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 100, "batch size")

flags.DEFINE_boolean('test', False, 'set True to test')
flags.DEFINE_boolean('build_data', False, 'set True to generate data')
flags.DEFINE_boolean('is_mtl', False, 'set True to use multi-task learning')
flags.DEFINE_boolean('is_adv', False, 'set True to use adv training')

FLAGS = tf.app.flags.FLAGS

def build_data():
  '''load raw data, build vocab, build TFRecord data, trim embeddings
  '''
  def _build_vocab(dbpedia_data, semeval_data):
    print('build vocab')
    dbpedia_vocab   = dbpedia.build_vocab(dbpedia_data)
    print('dbpedia vocab: %d' % len(dbpedia_vocab))
    semeval_vocab = semeval.build_vocab(semeval_data)
    print('semeval vocab: %d' % len(semeval_vocab))

    vocab = set(dbpedia_vocab)
    vocab.update(semeval_vocab)
    vocab = list(vocab)
    util.write_vocab(vocab)
    
  def _build_data(dbpedia_train, dbpedia_test, semeval_train, semeval_test):
    vocab2id = util.load_vocab2id()

    print('convert semeval data to TFRecord')
    semeval.write_as_tfrecord(semeval_train, semeval_test, vocab2id)
    print('convert dbpedia data to TFRecord')
    dbpedia.write_as_tfrecord(dbpedia_train, dbpedia_test, vocab2id)

  def _trim_embed():
    print('trimming pretrained embeddings')
    util.trim_embeddings(50)
    util.trim_embeddings(300)

  print('load raw data')
  # dbpedia_train, dbpedia_test = None, None
  dbpedia_train, dbpedia_test = dbpedia.load_raw_data(verbose=False)
  semeval_train, semeval_test = semeval.load_raw_data(verbose=False)

  # _build_vocab(dbpedia_train + dbpedia_test, semeval_train + semeval_test)

  _build_data(dbpedia_train, dbpedia_test, semeval_train, semeval_test)

  # _trim_embed()
  
def trace_runtime(sess, m_train):
  '''
  trace runtime bottleneck using timeline api

  navigate to the URL 'chrome://tracing' in a Chrome web browser, 
  click the 'Load' button and locate the timeline file.
  '''
  run_metadata=tf.RunMetadata()
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  from tensorflow.python.client import timeline
  trace_file = open('timeline.ctf.json', 'w')

  fetches = [m_train.train_op, m_train.loss, m_train.accuracy]
  _, loss, acc = sess.run(fetches, 
                            options=options, 
                            run_metadata=run_metadata)
                            
  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
  trace_file.write(trace.generate_chrome_trace_format())
  trace_file.close()

def train(sess, m_train, m_valid, semeval_test_iter, dbpedia_test_iter):
  best_acc, best_epoch = 0., 0
  start_time = time.time()
  orig_begin_time = start_time

  for epoch in range(FLAGS.num_epochs):
    sess.run([semeval_test_iter.initializer, dbpedia_test_iter.initializer])

    # train dbpedia and SemEval
    dbpedia_loss, dbpedia_acc = 0., 0.
    sem_loss, sem_acc = 0., 0.
    for batch in range(80):
      acc, loss,_ = m_train.tensors[0]
      train_op = m_train.train_ops[0]
      _, loss, acc = sess.run([train_op, loss, acc])
      dbpedia_loss += loss
      dbpedia_acc += acc
      
      acc, loss,_ = m_train.tensors[1]
      train_op = m_train.train_ops[1]
      _, loss, acc = sess.run([train_op, loss, acc])
      sem_loss += loss
      sem_acc += acc
      
    dbpedia_loss /= 80
    dbpedia_acc /= 80

    sem_loss /= 80
    sem_acc /= 80

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    sem_valid_acc = 0.
    dbpedia_valid_acc = 0.
    for batch in range(28):
    # while True:
    #   try:
      acc_tenser, _, _  = m_valid.tensors[0]
      acc = sess.run(acc_tenser)
      dbpedia_valid_acc += acc

      acc_tenser, _, _ = m_valid.tensors[1]
      acc = sess.run(acc_tenser)
      sem_valid_acc += acc
      # except tf.errors.OutOfRangeError:
      #   break
    sem_valid_acc /= 28
    dbpedia_valid_acc /= 28

    if best_acc < sem_valid_acc:
      best_acc = sem_valid_acc
      best_epoch = epoch
      m_train.save(sess, epoch)
    
    print("Epoch %d dbpedia %.2f %.2f %.4f sem %.2f %.2f %.4f time %.2f" % 
             (epoch, dbpedia_loss, dbpedia_acc, dbpedia_valid_acc, 
                      sem_loss, sem_acc, sem_valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_epoch, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def train_dbpedia(sess, m_train, m_valid, semeval_test_iter, dbpedia_test_iter):
  best_acc, best_epoch = 0., 0
  start_time = time.time()
  orig_begin_time = start_time

  for epoch in range(FLAGS.num_epochs):
    sess.run(dbpedia_test_iter.initializer)
    dbpedia_loss, dbpedia_acc = 0., 0.

    for batch in range(560):
      acc, loss,_ = m_train.tensors[0]
      train_op = m_train.train_ops[0]
      _, loss, acc = sess.run([train_op, loss, acc])
      dbpedia_loss += loss
      dbpedia_acc += acc

    dbpedia_loss /= 560
    dbpedia_acc /= 560

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    dbpedia_valid_acc = 0.
    for batch in range(170):
      acc_tenser, _, _ = m_valid.tensors[0]
      acc = sess.run(acc_tenser)
      dbpedia_valid_acc += acc
    dbpedia_valid_acc /= 170

    if best_acc < dbpedia_valid_acc:
      best_acc = dbpedia_valid_acc
      best_epoch = epoch
      m_train.save(sess, epoch)

    print("Epoch %d dbpedia %.2f %.2f %.4f time %.2f" % 
             (epoch, dbpedia_loss, dbpedia_acc, dbpedia_valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_epoch, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(sess, m_valid, semeval_test_iter):
  m_valid.restore(sess)
  sess.run([semeval_test_iter.initializer])
  acc_tenser, _, pred_tensor = m_valid.tensors[1]

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

  word_embed = util.load_embedding(word_dim=FLAGS.word_dim)
  with tf.Graph().as_default():
    semeval_train_iter, semeval_test_iter = semeval.read_tfrecord(
                                          FLAGS.num_epochs, FLAGS.batch_size)
    dbpedia_train_iter, dbpedia_test_iter = dbpedia.read_tfrecord(
                                          FLAGS.num_epochs, FLAGS.batch_size)
    model_name = 'mtl-dbpedia-%d' % FLAGS.word_dim
    semeval_train = semeval_train_iter.get_next()
    semeval_test = semeval_test_iter.get_next()
    dbpedia_train = dbpedia_train_iter.get_next()
    dbpedia_test = dbpedia_test_iter.get_next()
    m_train, m_valid = mtl_model.build_train_valid_model(
                                          model_name, word_embed, 
                                          semeval_train, semeval_test, 
                                          dbpedia_train, dbpedia_test, 
                                          FLAGS.is_mtl, FLAGS.is_adv)
    
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      print('='*80)

      if FLAGS.test:
        test(sess, m_valid, semeval_test_iter)
      else:
        # train_dbpedia(sess, m_train, m_valid, semeval_test_iter, dbpedia_test_iter)
        train(sess, m_train, m_valid, semeval_test_iter, dbpedia_test_iter)

if __name__ == '__main__':
  tf.app.run()
