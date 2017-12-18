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
flags.DEFINE_integer("num_epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 16, "batch size")

flags.DEFINE_boolean('test', False, 'set True to test')
flags.DEFINE_boolean('build_data', False, 'set True to generate data')

FLAGS = tf.app.flags.FLAGS

def build_data():
  '''load raw data, build vocab, build TFRecord data, trim embeddings
  '''
  def _build_vocab(all_data):
    print('build vocab')
    data = []
    for task_data in all_data:
      train_data, test_data = task_data
      data.extend(train_data + test_data)
    vocab = fudan.build_vocab(data)
    util.write_vocab(vocab)

    util.stat_length(data)
    
  def _build_data(all_data):
    print('build data')
    vocab2id = util.load_vocab2id()

    for task_id, task_data in enumerate(all_data):
      train_data, test_data = task_data
      fudan.write_as_tfrecord(train_data, test_data, task_id, vocab2id)

  def _trim_embed():
    print('trimming pretrained embeddings')
    util.trim_embeddings(50)
    util.trim_embeddings(300)

  print('load raw data')
  all_data = []
  for task_data in fudan.load_raw_data():
    all_data.append(task_data)
  
  _build_vocab(all_data)

  _build_data(all_data)
  _trim_embed()
  
def train(sess, models):
  n_models = len(models)
  
  best_acc, best_step= 0., 0
  start_time = time.time()
  orig_begin_time = start_time

  for epoch in range(FLAGS.num_epochs):
    all_loss, all_acc = 0., 0.
    for batch in range(82):
      for i in range(n_models):
        m_train, m_valid = models[i]
        train_fetch = [m_train.train, m_train.loss, m_train.acc]
        _, loss, acc = sess.run(train_fetch)
        all_loss += loss
        all_acc += acc

    all_loss /= (82*n_models)
    all_acc /= (82*n_models)

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    valid_acc = 0.
    for i in range(n_models):
      m_train, m_valid = models[i]
      acc = sess.run(m_valid.acc)
      valid_acc += acc
    valid_acc /= n_models

    if best_acc < valid_acc:
      best_acc = valid_acc
      best_step = epoch
      for i in range(n_models):
        m_train, m_valid = models[i]
        m_train.save(sess, epoch)
      
    print("Epoch %d loss %.2f acc %.2f %.4f time %.2f" % 
             (epoch, loss, acc, valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_step, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(sess, models):
  n_models = len(models)
  for i in range(n_models):
    task_name = fudan.get_task_name(i)
    m_train, m_valid = models[i]
    m_valid.restore(sess)
    acc = sess.run(m_valid.acc)
    print('%s acc: %.4f' % (task_name, acc))
  
def main(_):
  if FLAGS.build_data:
    build_data()
    exit()

  word_embed = util.load_embedding(word_dim=FLAGS.word_dim)
  with tf.Graph().as_default():
    models = []
    data_iter = fudan.read_tfrecord(FLAGS.num_epochs, FLAGS.batch_size)
    for task_id, (train_data, test_data) in enumerate(data_iter):
      task_name = fudan.get_task_name(task_id)
      model_name = 'task-%s-%d-%d' % (task_name, FLAGS.num_epochs, FLAGS.word_dim)
      m_train, m_valid = cnn_model.build_train_valid_model(
                                model_name, word_embed, train_data, test_data)
      models.append((m_train, m_valid))
      
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      print('='*80)

      if FLAGS.test:
        test(sess, models)
      else:
        train(sess, models)

if __name__ == '__main__':
  tf.app.run()
