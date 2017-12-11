import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import base
from inputs import imdb
from inputs import semeval
from models import cnn_model

# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_string("logdir", "saved_models/", "where to save the model")

flags.DEFINE_integer("num_relations", 19, "number of relations")
flags.DEFINE_integer("word_dim", 50, "word embedding size")
flags.DEFINE_integer("num_epochs", 200, "number of epochs")
flags.DEFINE_integer("batch_size", 100, "batch size")

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")

flags.DEFINE_float("lrn_rate", 1e-3, "learning rate")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

flags.DEFINE_boolean('test', False, 'set True to test')
flags.DEFINE_boolean('trace', False, 'set True to test')
flags.DEFINE_boolean('build_data', False, 'set True to generate data')

FLAGS = tf.app.flags.FLAGS

def build_data():
  '''load raw data, build vocab, build TFRecord data, trim embeddings
  '''
  print('load raw data')
  imdb_train, imdb_test       = imdb.load_raw_data()
  semeval_train, semeval_test = semeval.load_raw_data()

  print('build vocab')
  imdb_vocab   = imdb.build_vocab(imdb_train + imdb_test)
  emeval_vocab = semeval.build_vocab(semeval_train + semeval_test)
  vocab = imdb_vocab.union(emeval_vocab)
  base.write_vocab(vocab)
  del imdb_vocab
  del emeval_vocab
  del vocab

  vocab2id = base.load_vocab2id()
  print('convert semeval data to TFRecord')
  semeval.write_as_tfrecord(semeval_train, semeval_test, vocab2id)
  print('convert imdb data to TFRecord')
  imdb.write_as_tfrecord(imdb_train, imdb_test, vocab2id)
  del vocab2id

  print('trimming pretrained embeddings')
  base.trim_embeddings()

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
  n = 1
  best = .0
  best_step = n
  start_time = time.time()
  orig_begin_time = start_time

  fetches = [m_train.train_op, m_train.loss, m_train.accuracy]

  while True:
    try:
      _, loss, acc = sess.run(fetches)

      epoch = n // 80
      if n % 80 == 0:
        now = time.time()
        duration = now - start_time
        start_time = now
        v_acc = sess.run(m_valid.accuracy)
        if best < v_acc:
          best = v_acc
          best_step = n
          m_train.save(sess, best_step)
        print("Epoch %d, loss %.2f, acc %.2f %.4f, time %.2f" % 
                                  (epoch, loss, acc, v_acc, duration))
        sys.stdout.flush()
      n += 1
    except tf.errors.OutOfRangeError:
      break

  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_step: %d, best_acc: %.4f' % (best_step, best))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(sess, m_valid):
  m_valid.restore(sess)
  fetches = [m_valid.accuracy, m_valid.prediction]
  accuracy, predictions = sess.run(fetches)
  print('accuracy: %.4f' % accuracy)
  
  base_reader.write_results(predictions, FLAGS.relations_file, FLAGS.results_file)


def main(_):
  if FLAGS.build_data:
    build_data()
    exit()

  with tf.Graph().as_default():
    word_embed = base.load_embedding()
    semeval_train, semeval_test = semeval.read_tfrecord(
                                                FLAGS.num_epochs, 
                                                FLAGS.batch_size)
    imdb_train = imdb.read_tfrecord(FLAGS.num_epochs, FLAGS.batch_size)

    m_train, m_valid = cnn_model.build_train_valid_model(word_embed, 
                                                      train_data, test_data)
    
    m_train.set_saver('cnn-%d-%d' % (FLAGS.num_epochs, FLAGS.word_dim))
    
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
    config.gpu_options.allow_growth = True
    
    # sv finalize the graph
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
