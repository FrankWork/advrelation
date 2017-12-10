import os
import tensorflow as tf
from collections import defaultdict
from collections import namedtuple

from inputs import base

flags = tf.app.flags

flags.DEFINE_string("imdb_train_dir", "data/aclImdb/train", 
                             "imdb training directory")
flags.DEFINE_string("imdb_test_dir", "data/aclImdb/test", 
                             "imdb test directory")

FLAGS = flags.FLAGS


Raw_Example = namedtuple('Raw_Example', 'label sentence')

def _load_raw_data_from_dir(dir, neg_or_pos):
  assert neg_or_pos in ('neg', 'pos')

  data = []
  dir = os.path.join(dir, neg_or_pos)
  label = True if neg_or_pos == 'pos' else False

  for filename in os.listdir(dir):
    filename = os.path.join(dir, filename)
    with open(filename) as f:
      lines = f.readlines()
      assert len(lines) == 1
      # line = base.clean_str(lines[0])
      # tokens = line.split(' ')
      tokens = base.split_by_punct(lines[0])

      example = Raw_Example(label, tokens)
      data.append(example)
  return data

def _load_neg_pos_data(dir):
  pos_data = _load_raw_data_from_dir(dir, 'pos')
  neg_data = _load_raw_data_from_dir(dir, 'neg')
  data = pos_data + neg_data
  return data

def load_imdb_raw_data():
  train_data = _load_neg_pos_data(FLAGS.imdb_train_dir)
  test_data = _load_neg_pos_data(FLAGS.imdb_test_dir)
  return train_data, test_data

if __name__ == '__main__':

  load_imdb_data("data/aclImdb/train")