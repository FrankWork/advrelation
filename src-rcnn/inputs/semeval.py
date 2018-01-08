import re
import os
import tensorflow as tf
from collections import namedtuple, defaultdict

from inputs import util

flags = tf.app.flags

flags.DEFINE_string("semeval_train_file", "data/SemEval/train.cln", 
                             "original training file")
flags.DEFINE_string("semeval_test_file", "data/SemEval/test.cln", 
                             "original test file")

flags.DEFINE_string("semeval_train_record", 
                              "data/generated/train.semeval.tfrecord", 
                             "training file of TFRecord format")
flags.DEFINE_string("semeval_test_record", 
                             "data/generated/test.semeval.tfrecord", 
                             "Test file of TFRecord format")

flags.DEFINE_string("relations_file", "data/SemEval/relations.txt", 
                             "relations file, map relation id to its text")
flags.DEFINE_string("results_file", "data/generated/results.txt", 
                             "predicted results file")

FLAGS = flags.FLAGS

OUTPUT_DIR = "data/generated"
VOCAB_FILE = os.path.join(OUTPUT_DIR, "semeval.vocab")
VOCAB_FREQ_FILE = os.path.join(OUTPUT_DIR, "semeval.vocab.freq")

MAX_VOCAB_SIZE = 2**13
Raw_Example = namedtuple('Raw_Example', 'label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair', 'first last')

def _load_relations():
  id2relation = []
  relation2id = dict()

  with open(FLAGS.relations_file) as f:
    for id, line in enumerate(f):
      rel = line.strip()
      id2relation.append(rel)
      relation2id[rel] = id
  
  return relation2id, id2relation

def _load_raw_data(filename):
  '''load raw data from text file, 

  return: a list of Raw_Example
  '''
  data = []
  with open(filename) as f:
    for line in f:
      words = line.strip().split(' ')
      
      sent = words[5:]

      label = int(words[0])

      entity1 = PositionPair(int(words[1]), int(words[2]))
      entity2 = PositionPair(int(words[3]), int(words[4]))

      example = Raw_Example(label, entity1, entity2, sent)
      data.append(example)
  return data

def _length_statistics(raw_train_data, raw_test_data):
  '''statistics info of the length'''
  print('SemEval sentence length:')
  length1 = [len(example.sentence) for example in raw_train_data]
  length2 = [len(example.sentence) for example in raw_test_data]
  util.length_statistics(length1 + length2)

def load_raw_data(verbose=False):
  # relation2id, _ = _load_relations()
  train_data = _load_raw_data(FLAGS.semeval_train_file)
  test_data = _load_raw_data(FLAGS.semeval_test_file)
  if verbose:
    _length_statistics(train_data, test_data)
  return train_data, test_data

def build_vocab(raw_data):
  '''collect words in sentence'''
  vocab_freqs = defaultdict(int)

  for example in raw_data:
    tokens = example.sentence
    for token in tokens:
      vocab_freqs[token] += 1
  
  # Filter out low-occurring terms
  vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items())
                      # if vocab_freqs[term] > 1)

  # Sort by frequency
  ordered_vocab_freqs = sorted(
      vocab_freqs.items(), key=lambda item: item[1], reverse=True)

  # Limit vocab size
  # ordered_vocab_freqs = ordered_vocab_freqs[:MAX_VOCAB_SIZE]

  # vocab = [token for token, _ in ordered_vocab_freqs]
  # util.write_vocab(vocab, FLAGS.semeval_vocab_file)
  # return vocab
  util.write_vocab_and_freq(ordered_vocab_freqs, VOCAB_FILE, VOCAB_FREQ_FILE)
  # return ordered_vocab_freqs

def load_vocab_and_freq():
  return util.load_voab_and_freq(VOCAB_FILE, VOCAB_FREQ_FILE)

def load_vocab2id():
  return util.load_vocab2id(VOCAB_FILE)

def _map_tokens_and_pad(raw_example, vocab2id):
  '''inplace map tokens in raw_example to ids
  Args:
    raw_example: type Raw_Example._asdict()
    vocab2id: dict<token, id>
  '''
  sentence = util.map_tokens_to_ids(raw_example['sentence'], vocab2id)
  raw_example['sentence'] = sentence
  # raw_example['sentence'] = util.pad_or_truncate(sentence, FLAGS.semeval_max_len)

def _position_feature(raw_example):
  e1_idx = raw_example['entity1'].first
  e2_idx = raw_example['entity2'].first

  position1 = []
  position2 = []
  length = len(raw_example['sentence'])
  for i in range(length):
    position1.append(util.relative_distance(i-e1_idx))
    position2.append(util.relative_distance(i-e2_idx))
  
  return position1, position2

def _build_sequence_example(raw_example):
  '''build tf.train.SequenceExample from Raw_Example
  context features : lexical, rid
  sequence features: sentence, position1, position2

  Args: 
    raw_example : type Raw_Example._asdict()

  Returns:
    tf.trian.SequenceExample
  '''
  ex = tf.train.SequenceExample()

  label = raw_example['label']
  ex.context.feature['label'].int64_list.value.append(label)

  sentence = raw_example['sentence']
  ex.context.feature['length'].int64_list.value.append(len(sentence))

  for word_id in sentence:
    word = ex.feature_lists.feature_list['sentence'].feature.add()
    word.int64_list.value.append(word_id)
  

  # relative distance from entity1, entity2
  position1, position2 = _position_feature(raw_example)
  for pos_id in position1:
    pos = ex.feature_lists.feature_list['position1'].feature.add()
    pos.int64_list.value.append(pos_id)
  for pos_id in position2:
    pos = ex.feature_lists.feature_list['position2'].feature.add()
    pos.int64_list.value.append(pos_id)
  
  return ex


def write_as_tfrecord(train_data, test_data, vocab2id):
  '''convert the raw data to TFRecord format and write to disk
  '''
  util.write_as_tfrecord(train_data, 
                         vocab2id, 
                         FLAGS.semeval_train_record, 
                         _map_tokens_and_pad, 
                         _build_sequence_example)
  util.write_as_tfrecord(test_data, 
                         vocab2id, 
                         FLAGS.semeval_test_record, 
                         _map_tokens_and_pad, 
                         _build_sequence_example)

  util.shuf_and_write(FLAGS.semeval_train_record)

def _parse_tfexample(serialized_example):
  '''parse serialized tf.train.SequenceExample to tensors
  context features : lexical, label
  sequence features: sentence, position1, position2
  '''
  context_features={
                      'label'    : tf.FixedLenFeature([], tf.int64),
                      'length'    : tf.FixedLenFeature([], tf.int64)}
  sequence_features={
                      'sentence' : tf.FixedLenSequenceFeature([], tf.int64),
                      'position1'  : tf.FixedLenSequenceFeature([], tf.int64),
                      'position2'  : tf.FixedLenSequenceFeature([], tf.int64),}
  context_dict, sequence_dict = tf.parse_single_sequence_example(
                      serialized_example,
                      context_features   = context_features,
                      sequence_features  = sequence_features)
  
  label = context_dict['label']
  length = context_dict['length']

  sentence  = sequence_dict['sentence']
  position1 = sequence_dict['position1']
  position2 = sequence_dict['position2']

  return (label, length, sentence, position1, position2)

def read_tfrecord(epoch, batch_size):
  padded_shapes = ([], [], [98], [98], [98])

  train_data = util.read_tfrecord(FLAGS.semeval_train_record, 
                              epoch, 
                              batch_size, 
                              _parse_tfexample,
                              padded_shapes,
                              shuffle=True)
  test_data = util.read_tfrecord(FLAGS.semeval_test_record, 
                              epoch, 
                              batch_size, 
                              _parse_tfexample,
                              padded_shapes,
                              shuffle=False)

  return train_data, test_data

def write_results(predictions):
  _, id2relations = _load_relations()
  
  start_no = 8001
  with open(FLAGS.results_file, 'w') as f:
    for idx, id in enumerate(predictions):
      if idx < 2717:
        rel = id2relations[id]
        f.write('%d\t%s\n' % (start_no+idx, rel))
