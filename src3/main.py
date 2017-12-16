import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import util
from inputs import fudan
# from inputs import imdb
# from inputs import semeval
# from models import mtl_model
from models import cnn_model
# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_integer("word_dim", 300, "word embedding size")
flags.DEFINE_integer("num_epochs", 200, "number of epochs")
flags.DEFINE_integer("batch_size", 50, "batch size")

flags.DEFINE_boolean('test', False, 'set True to test')
flags.DEFINE_boolean('trace', False, 'set True to trace runtime bottleneck')
flags.DEFINE_boolean('build_data', False, 'set True to generate data')

FLAGS = tf.app.flags.FLAGS

def build_data():
  '''load raw data, build vocab, build TFRecord data, trim embeddings
  '''
  def _build_vocab(all_data):
    print('build vocab')
    for task_data in all_data:
      train_data, test_data = task_data
      data.extend(train_data + test_data)
    vocab = fudan.build_vocab(data)
    util.write_vocab(vocab)
    
  def _build_data(all_data):
    vocab2id = util.load_vocab2id()

    for task_id, task_data in enumerate(all_data):
      train_data, test_data = task_data
      fudan.write_as_tfrecord(train_data, test_data, task_id, vocab2id)

  def _trim_embed():
    print('trimming pretrained embeddings')
    util.trim_embeddings(FLAGS.word_dim)

  print('load raw data')
  all_data = []
  for task_data in fudan.load_raw_data():
    all_data.append(task_data)
  
  _build_vocab(all_data)

  _build_data(all_data)
  _trim_embed()
  
def train(sess, m_train, m_valid):
  imdb_train_fetch =    [m_train.imdb_train, 
                         m_train.imdb_loss, m_train.imdb_accuracy]
  semeval_train_fetch = [m_train.semeval_train, 
                         m_train.semeval_loss, m_train.semeval_accuracy]

  best_acc, best_step= 0., 0
  start_time = time.time()
  orig_begin_time = start_time

  for epoch in range(FLAGS.num_epochs):
    # train imdb and SemEval
    imdb_loss, imdb_acc = 0., 0.
    sem_loss, sem_acc = 0., 0.
    for batch in range(160):
      _, loss, acc = sess.run(imdb_train_fetch)
      imdb_loss += loss
      imdb_acc += acc

      _, loss, acc = sess.run(semeval_train_fetch)
      sem_loss += loss
      sem_acc += acc

    imdb_loss /= 160
    imdb_acc /= 160

    sem_loss /= 160
    sem_acc /= 160

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    # imdb_valid_acc = 0
    imdb_valid_acc = sess.run(m_valid.imdb_accuracy)
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

  word_embed = util.load_embedding(word_dim=FLAGS.word_dim)
  with tf.Graph().as_default():
    data_iter = fudan.read_tfrecord(FLAGS.num_epochs, FLAGS.batch_size)
    for task_id, train_data, test_data in enumerate(data_iter):
      task_name = fudan.get_task_name(task_id)
      model_name = '%s-%d-%d' % (task_name, FLAGS.num_epochs, FLAGS.word_dim)
      m_train, m_valid = cnn_model.build_train_valid_model(
                                model_name, word_embed, train_data, test_data)
      
      
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
