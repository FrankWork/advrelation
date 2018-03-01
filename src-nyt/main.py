import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import  dataset, rc_dataset, utils
from models import cnn_model
import config as config_lib

# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags
flags.DEFINE_boolean('test', False, 'set True to test')
FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def train_semeval(config, session, m_train, m_valid, test_iter):
  best_acc, best_epoch = 0., 0
  start_time = time.time()
  orig_begin_time = start_time
  
  for epoch in range(config.hparams.num_epochs):
    loss, acc = m_train.train_epoch(session, 80)

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    valid_acc = m_valid.evaluate(session, test_iter, 28)

    if best_acc < valid_acc:
      best_acc = valid_acc
      best_epoch = epoch
      m_train.save(session, epoch)
    
    print("Epoch %d train %.2f %.2f valid %.2f time %.2f" % 
             (epoch, loss, acc, valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_epoch, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(config, session, m_valid, test_iter):
  m_valid.restore(session)
  all_pred = m_valid.pred_results(session, test_iter, 28)
  utils.write_results(all_pred, "data/SemEval/relations.txt", "data/generated/results.txt")

def main(_):
  config = config_lib.get_config()
  embed = dataset.Embed(config.out_dir, config.trimmed_embed300_file, config.vocab_file)
  ini_word_embed = embed.load_embedding()

  record_data = rc_dataset.RCRecordData(config.out_dir, 
                config.train_record, config.semeval_test_record)

  with tf.Graph().as_default():
    train_iter = record_data.train_data(config.hparams.num_epochs, config.hparams.batch_size)
    test_iter = record_data.test_data(1, config.hparams.batch_size)

                                          
    train_data = train_iter.get_next()
    test_data = test_iter.get_next()

    m_train, m_valid = cnn_model.build_train_valid_model(config, 
                                          ini_word_embed, train_data, test_data)

    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    for tensor in tf.trainable_variables():
      tf.logging.info(tensor.op.name)
        
    with tf.Session(config=sess_config) as sess:
      sess.run(init_op)
      print('='*80)

      if FLAGS.is_test:
        test(config, sess, m_valid, test_iter)
      else:
        train_semeval(config, sess, m_train, m_valid, test_iter)

if __name__ == '__main__':
  tf.app.run()
