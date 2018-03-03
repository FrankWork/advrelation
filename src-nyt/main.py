import os
import time
import sys
import math
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


def train(session, m_train, m_valid, test_iter, restore=False):
  if restore:
    m_train.restore(session)

  print('='*80)

  best_acc, best_step = 0., 0
  global_step_tensor = tf.train.get_global_step()

  start_time = time.time()
  orig_begin_time = start_time
  
  hparams = m_train.hparams
  num_batches_train = math.ceil(hparams.num_train_examples / hparams.batch_size)
  num_batches_test  = math.ceil(hparams.num_test_examples / hparams.batch_size)

  batch = 0
  moving_acc = []
  moving_loss = []
  while True:
    try:
      loss, acc = m_train.train_step(session)
      moving_loss.append(loss)
      moving_acc.append(acc)

      batch += 1
      if batch % hparams.log_freq == 0:
        # duration
        now = time.time()
        duration = now - start_time
        start_time = now

        # log
        loss = np.mean(moving_loss)
        acc = np.mean(moving_acc)*100
        print("Epoch %d batch %d loss %.2f acc %.2f time %.2f" % 
              (batch/num_batches_train, batch, loss, acc, duration))
        sys.stdout.flush()
        moving_loss.clear()
        moving_acc.clear()
      
      if batch % num_batches_train == 0:
        # valid accuracy
        valid_acc = m_valid.evaluate(session, test_iter, num_batches_test)

        if best_acc < valid_acc:
          best_acc = valid_acc
          best_step = tf.train.global_step(session, global_step_tensor)
          m_train.save(session, best_step)
        print('\t Valid acc: %.2f' % valid_acc)
        sys.stdout.flush()

    except tf.errors.OutOfRangeError:
      break
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  # print('Done training, best_epoch: %d, best_acc: %.4f' % (best_epoch, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(session, m_valid, test_iter):
  m_valid.restore(session)
  all_pred = m_valid.pred_results(session, test_iter, 28)
  utils.write_results(all_pred, "data/SemEval/relations.txt", "data/generated/results.txt")

def main(_):
  config = config_lib.get_config()
  embed = dataset.Embed(config.out_dir, config.trimmed_embed300_file, config.vocab_file)
  ini_word_embed = embed.load_embedding()

  semeval_data = rc_dataset.RCRecordData(config.out_dir, 
                config.semeval_train_record, config.semeval_test_record)
  nyt_data = rc_dataset.RCRecordData(config.out_dir, 
                config.nyt_train_record, config.nyt_test_record)
  # nyt_data.count_records()

  semeval_hparams = config_lib.semeval_hparams()
  nyt_hparams = config_lib.nyt_hparams()

  with tf.Graph().as_default():
    semeval_train_iter = semeval_data.train_data(semeval_hparams.num_epochs, 
                                                 semeval_hparams.batch_size)
    semeval_test_iter = semeval_data.test_data(1, semeval_hparams.batch_size)

    nyt_train_iter = nyt_data.train_data(nyt_hparams.num_epochs, 
                                         nyt_hparams.batch_size)
    nyt_test_iter = nyt_data.test_data(1, nyt_hparams.batch_size)

    semeval_train_data = semeval_train_iter.get_next()
    semeval_test_data = semeval_test_iter.get_next()
    nyt_train_data = nyt_train_iter.get_next()
    nyt_test_data = nyt_test_iter.get_next()

    sem_train, sem_valid = cnn_model.build_train_valid_model(semeval_hparams, 
                  ini_word_embed, semeval_train_data, semeval_test_data)
    nyt_train, nyt_valid = cnn_model.build_train_valid_model(nyt_hparams, 
                  ini_word_embed, nyt_train_data, nyt_test_data)

    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    for tensor in tf.trainable_variables():
      tf.logging.info(tensor.op.name)
        
    with tf.Session(config=sess_config) as sess:
      sess.run(init_op)

      if FLAGS.test:
        test(sess, sem_valid, semeval_test_iter)
      else:
        # train(sess, nyt_train, nyt_valid, nyt_test_iter)
        train(sess, sem_train, sem_valid, semeval_test_iter, restore=True)

if __name__ == '__main__':
  tf.app.run()
