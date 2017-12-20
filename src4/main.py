import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import util
from inputs import fudan
# from inputs import imdb
# from inputs import semeval
from models import mtl_model
# from models import cnn_model
# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_integer("word_dim", 300, "word embedding size")
flags.DEFINE_integer("num_epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 16, "batch size")

flags.DEFINE_boolean('test', False, 'set True to test')
flags.DEFINE_boolean('build_data', False, 'set True to generate data')
flags.DEFINE_boolean('adv', False, 'set True to use adv training')

FLAGS = tf.app.flags.FLAGS

def build_data():
  '''load raw data, build vocab, build TFRecord data, trim embeddings
  '''
  def _build_vocab(data):
    print('build vocab')
    vocab = fudan.build_vocab(data)
    util.write_vocab(vocab)

    util.stat_length(data)
    
  def _build_data(train_data, test_data):
    print('build data')
    vocab2id = util.load_vocab2id()

    fudan.write_as_tfrecord(train_data, test_data, vocab2id)

  def _trim_embed():
    print('trimming pretrained embeddings')
    util.trim_embeddings(50)
    util.trim_embeddings(300)

  print('load raw data')
  train_data, test_data = fudan.load_raw_data()
  
  _build_vocab(train_data + test_data)

  _build_data(train_data, test_data)
  _trim_embed()
  
def train(sess, m_train, m_valid, test_iter):
  train_fetch = [m_train.train, m_train.loss, m_train.acc, m_train.adv_acc]
  
  best_acc, best_step= 0., 0
  start_time = time.time()
  orig_begin_time = start_time

  for epoch in range(FLAGS.num_epochs):
    sess.run(test_iter.initializer)

    all_loss, all_acc, all_adv_acc = 0., 0., 0.
    for batch in range(1386):
      _, loss, acc, adv_acc = sess.run(train_fetch)
      all_loss += loss
      all_acc += acc
      all_adv_acc += adv_acc

    all_loss /= 1386
    all_acc /= 1386
    all_adv_acc /= 1386

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    valid_acc = 0.
    while True:
      try:
        acc = sess.run(m_valid.acc)
        valid_acc += acc
      except tf.errors.OutOfRangeError:
        break
    valid_acc /= 350
    
    if best_acc < valid_acc:
      best_acc = valid_acc
      best_step = epoch
      m_train.save(sess, epoch)
    print("Epoch %d loss %.2f adv_acc %.4f acc %.2f %.4f time %.2f" % 
             (epoch, all_loss, all_adv_acc, all_acc, valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_step, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()
  
def main(_):
  if FLAGS.build_data:
    build_data()
    exit()

  word_embed = util.load_embedding(word_dim=FLAGS.word_dim)
  with tf.Graph().as_default():
    train_iter, test_iter = fudan.read_tfrecord(FLAGS.num_epochs, 
                                                FLAGS.batch_size)
    train_data = train_iter.get_next()
    test_data = test_iter.get_next()

    m_train, m_valid = mtl_model.build_train_valid_model(
                    'fudan-mtl', word_embed, train_data, test_data, FLAGS.adv)
      
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
        train(sess, m_train, m_valid, test_iter)

if __name__ == '__main__':
  tf.app.run()
