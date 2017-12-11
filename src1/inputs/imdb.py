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

flags.DEFINE_string("imdb_train_record", "data/generated/train.imdb.tfrecord", 
                             "training file of TFRecord format")
flags.DEFINE_string("imdb_test_record", "data/generated/test.imdb.tfrecord", 
                             "Test file of TFRecord format")

flags.DEFINE_integer('doc_count_threshold', 1, 'The minimum number of '
                     'documents a word or bigram should occur in to keep '
                     'it in the vocabulary.')
FLAGS = flags.FLAGS

MAX_VOCAB_SIZE = 100 * 1000
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
      tokens = base.wordpunct_tokenizer(lines[0].lower())

      example = Raw_Example(label, tokens)
      data.append(example)
  return data

def _load_neg_pos_data(dir):
  pos_data = _load_raw_data_from_dir(dir, 'pos')
  neg_data = _load_raw_data_from_dir(dir, 'neg')
  data = pos_data + neg_data
  return data

def load_raw_data():
  train_data = _load_neg_pos_data(FLAGS.imdb_train_dir)
  test_data = _load_neg_pos_data(FLAGS.imdb_test_dir)
  return train_data, test_data

def build_vocab(raw_data):
  vocab_freqs = defaultdict(int)
  doc_counts = defaultdict(int)

  for example in raw_data:
    doc = example.sentence
    doc_seen = set()
    for token in doc:
      vocab_freqs[token] += 1
      if token not in doc_seen:
        doc_counts[token] += 1
        doc_seen.add(token)
  
  # Filter out low-occurring terms
  vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items()
                      if doc_counts[term] > FLAGS.doc_count_threshold)

  # Sort by frequency
  ordered_vocab_freqs = sorted(
      vocab_freqs.items(), key=lambda item: item[1], reverse=True)

  # Limit vocab size
  # ordered_vocab_freqs = ordered_vocab_freqs[:MAX_VOCAB_SIZE]

  vocab = [token for token, _ in ordered_vocab_freqs]
  vocab = set(vocab)
  return vocab

def _build_sequence_example(raw_example):
  '''build tf.train.SequenceExample from Raw_Example
  context features : label
  sequence features: sentence

  Args: 
    raw_example : type Raw_Example

  Returns:
    tf.trian.SequenceExample
  '''
  ex = tf.train.SequenceExample()

  label = raw_example.label
  ex.context.feature['label'].int64_list.value.append(label)

  for word_id in raw_example.sentence:
    word = ex.feature_lists.feature_list['sentence'].feature.add()
    word.int64_list.value.append(word_id)
  
  return ex

def _write_as_tfrecord(raw_data, vocab2id, filename):
  '''convert the raw data to TFRecord format and write to disk
  '''
  base.map_tokens_to_id(raw_data, vocab2id)
  records = []
  for raw_example in raw_data:
    example = _build_sequence_example(raw_example)
    records.append(example)
  del raw_data
  
  base.write_tfrecord(records, filename)

def write_as_tfrecord(train_data, test_data, vocab2id):
  '''convert the raw data to TFRecord format and write to disk
  '''
  _write_as_tfrecord(train_data, vocab2id, FLAGS.imdb_train_record)
  _write_as_tfrecord(test_data, vocab2id, FLAGS.imdb_test_record)
