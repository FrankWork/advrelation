import os
from collections import namedtuple, defaultdict

from inputs import util

DBPEDIA_TRAIN_FILE = "data/dbpedia_csv/train.csv"
DBPEDIA_TEST_FILE = "data/dbpedia_csv/test.csv"
DBPEDIA_VOCAB_FILE = "data/generated/vocab.dbpedia.txt"
DBPEDIA_TRAIN_RECORD = "data/generated/train.dbpedia.tfrecord"
DBPEDIA_TEST_RECORD = "data/generated/test.dbpedia.tfrecord"
DBPEDIA_MAX_LEN = 0

Raw_Example = namedtuple('Raw_Example', 'label entity sentence')

def _pad_or_truncate_entity(tokens):
  tokens = tokens[:3]

  pad_n = 3 - len(tokens)
  tokens.extend(pad_n*[util.PAD_WORD])

def _load_raw_data(filename):
  '''load raw data from text file, 
  file contents:
    label,"entity","sentence"
    ...
    EOF

  return: a list of Raw_Example
  '''
  data = []
  with open(filename) as f:
    for line in f:
      segments = line.strip().lower().split(",", 2)
      assert len(segments) == 3
      
      label = int(segments[0])

      entity   = util.split_by_punct( segments[1].strip('"') )
      entity = _pad_or_truncate_entity(entity)

      sentence = util.wordpunct_tokenizer( segments[2].strip('"') )

      example = Raw_Example(label, entity, sentence)
      data.append(example)

  return data

def _stat_entity_length(raw_data):
  '''get max_len and avg_len from data
  '''
  import numpy as np
  length = [len(example.entity) for example in raw_data]
  length = sorted(length)
  length = np.asarray(length)

  max_len = np.max(length)
  avg_len = np.mean(length)
  med_len = np.median(length)
  print('entity max_len: %d, avg_len: %d, med_len: %d' %(max_len, avg_len, med_len))

def load_raw_data():
  train_data = _load_raw_data(DBPEDIA_TRAIN_FILE)
  test_data = _load_raw_data(DBPEDIA_TEST_FILE)
  # _stat_entity_length(train_data)
  # _stat_entity_length(test_data)
  return train_data, test_data

def build_vocab(raw_data):
  '''collect words in sentence'''
  vocab = set()
  for example in raw_data:
    for w in example.sentence:
        vocab.add(w)
  
  util.write_vocab(vocab, DBPEDIA_VOCAB_FILE)
  return vocab

def _build_sequence_example(raw_example):
  '''build tf.train.SequenceExample from Raw_Example
  context features : label
  sequence features: sentence, entity

  Args: 
    raw_example : type Raw_Example._asdict()

  Returns:
    tf.trian.SequenceExample
  '''
  ex = tf.train.SequenceExample()

  label = raw_example['label']
  ex.context.feature['label'].int64_list.value.append(label)

  for word_id in raw_example['sentence']:
    word = ex.feature_lists.feature_list['sentence'].feature.add()
    word.int64_list.value.append(word_id)
  
  for word_id in raw_example['entity']:
    word = ex.feature_lists.feature_list['entity'].feature.add()
    word.int64_list.value.append(word_id)

  return ex

def _map_tokens_to_ids(raw_example, vocab2id):
  '''inplace convert sentence from a list of tokens to a list of ids
  Args:
    raw_example: an instance of Raw_Example._asdict()
    vocab2id: dict<token, id> {token0: id0, ...}
  '''
  sent_id = []
  for token in raw_example['sentence']:
    if token in vocab2id:
      tok_id = vocab2id[token]
      sent_id.append(tok_id)
  raw_example['sentence'] = sent_id
   
def write_as_tfrecord(train_data, test_data, vocab2id):
  '''convert the raw data to TFRecord format and write to disk
  '''
  util.write_as_tfrecord(train_data, 
                         vocab2id, 
                         DBPEDIA_TRAIN_RECORD, 
                         DBPEDIA_MAX_LEN, 
                         _build_sequence_example)
  util.write_as_tfrecord(test_data, 
                         vocab2id, 
                         DBPEDIA_TEST_RECORD, 
                         DBPEDIA_MAX_LEN, 
                         _build_sequence_example)

  util.shuf_and_write(DBPEDIA_TRAIN_RECORD)

def _parse_tfexample(serialized_example):
  '''parse serialized tf.train.SequenceExample to tensors
  context features : label
  sequence features: sentence, entity
  '''
  context_features={'label'    : tf.FixedLenFeature([], tf.int64)}
  sequence_features={'sentence' : tf.FixedLenSequenceFeature([], tf.int64),
                     'entity'  : tf.FixedLenSequenceFeature([], tf.int64)}
  context_dict, sequence_dict = tf.parse_single_sequence_example(
                      serialized_example,
                      context_features   = context_features,
                      sequence_features  = sequence_features)

  sentence = sequence_dict['sentence']
  entity = sequence_dict['entity']

  label = context_dict['label']

  return label, sentence, entity

def read_tfrecord(epoch, batch_size):
  train_data = util.read_tfrecord(DBPEDIA_TRAIN_RECORD, 
                                    epoch, 
                                    batch_size, 
                                    _parse_tfexample,
                                    shuffle=True)
  test_data = util.read_tfrecord(DBPEDIA_TEST_RECORD, 
                                    epoch, 
                                    batch_size, 
                                    _parse_tfexample,
                                    shuffle=False)

  return train_data, test_data

