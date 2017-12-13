import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import util
from inputs import imdb
from inputs import semeval
from models import mtl_model

# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_string("logdir", "saved_models/", "where to save the model")


flags.DEFINE_integer("word_dim", 50, "word embedding size")
flags.DEFINE_integer("num_epochs", 200, "number of epochs")
flags.DEFINE_integer("batch_size", 100, "batch size")

flags.DEFINE_boolean('test', False, 'set True to test')
flags.DEFINE_boolean('trace', False, 'set True to trace runtime bottleneck')
flags.DEFINE_boolean('build_data', False, 'set True to generate data')

FLAGS = tf.app.flags.FLAGS

def build_data():
  '''load raw data, build vocab, build TFRecord data, trim embeddings
  '''
  def _build_vocab(imdb_data, semeval_data):
    print('build vocab')
    imdb_vocab   = imdb.build_vocab(imdb_data)
    print('imdb vocab: %d' % len(imdb_vocab))
    semeval_vocab = semeval.build_vocab(semeval_data)
    print('semeval vocab: %d' % len(semeval_vocab))
    inter_vocab = imdb_vocab.intersection(semeval_vocab)
    print('imdb semeval intersection vocab: %d' % len(inter_vocab)) # 16717
    union_vocab = imdb_vocab.union(semeval_vocab)
    print('imdb semeval union vocab: %d' % len(union_vocab)) # 65189

    util.write_vocab(union_vocab)

  def _build_data(imdb_train, imdb_test, semeval_train, semeval_test):
    vocab2id = util.load_vocab2id()

    print('convert semeval data to TFRecord')
    semeval.write_as_tfrecord(semeval_train, semeval_test, vocab2id)
    print('convert imdb data to TFRecord')
    imdb.write_as_tfrecord(imdb_train, imdb_test, vocab2id)

  def _trim_embed():
    print('trimming pretrained embeddings')
    util.trim_embeddings()

  print('load raw data')
  imdb_train, imdb_test       = imdb.load_raw_data()
  semeval_train, semeval_test = semeval.load_raw_data()
  
  _build_vocab(imdb_train + imdb_test, semeval_train + semeval_test)
  _build_data(imdb_train, imdb_test, semeval_train, semeval_test)
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

def train(sess, m_train, m_valid):
  imdb_train_fetch =    [m_train.imdb_train, 
                         m_train.imdb_loss, m_train.imdb_accuracy]
  semeval_train_fetch = [m_train.semeval_train, 
                         m_train.semeval_loss, m_train.semeval_accuracy]

  best_acc, best_step= 0., 0
  start_time = time.time()
  orig_begin_time = start_time

  for epoch in range(FLAGS.num_epochs):
    # train imdb
    imdb_loss, imdb_acc = 0., 0.
    for batch in range(250):
      _, loss, acc = sess.run(imdb_train_fetch)
      imdb_loss += loss
      imdb_acc += acc
    imdb_loss /= 250
    imdb_acc /= 250

    imdb_valid_acc = sess.run(m_valid.imdb_accuracy)

    # train SemEval
    sem_loss, sem_acc = 0., 0.
    for batch in range(80):
      _, loss, acc = sess.run(semeval_train_fetch)
      sem_loss += loss
      sem_acc += acc
    sem_loss /= 80
    sem_acc /= 80

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    sem_valid_acc = sess.run(m_valid.semeval_accuracy)
    if best_acc < sem_valid_acc:
      best_acc = sem_valid_acc
      best_step = sess.run(m_train.global_step)
      m_train.save(sess, best_step)
    
    print("Epoch %d imdb %.2f %.2f %.4f sem %.2f %.2f %.4f time %.2f" % 
             (epoch, imdb_loss, imdb_acc, imdb_valid_acc, 
                      sem_loss, sem_acc, sem_valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_step: %d, best_acc: %.4f' % (best_step, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(sess, m_valid):
  m_valid.restore(sess)
  fetches = [m_valid.semeval_accuracy, m_valid.semeval_pred]
  accuracy, predictions = sess.run(fetches)
  print('accuracy: %.4f' % accuracy)
  
  semeval.write_results(predictions)

def main(_):
  if FLAGS.build_data:
    build_data()
    exit()

  word_embed = util.load_embedding()
  with tf.Graph().as_default():
    semeval_train, semeval_test = semeval.read_tfrecord(
                                          FLAGS.num_epochs, FLAGS.batch_size)
    imdb_train, imdb_test = imdb.read_tfrecord(
                                          FLAGS.num_epochs, FLAGS.batch_size)
      
    m_train, m_valid = mtl_model.build_train_valid_model(
                                          word_embed, 
                                          semeval_train, semeval_test, 
                                          imdb_train, imdb_test)
    m_train.set_saver('mtl-%d-%d' % (FLAGS.num_epochs, FLAGS.word_dim))
    m_train.build_train_op()
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      print('='*80)

      if FLAGS.trace:
        trace_runtime(sess, m_train)
      elif FLAGS.test:
        test(sess, m_valid)
      else:
        train(sess, m_train, m_valid)

if __name__ == '__main__':
  tf.app.run()
