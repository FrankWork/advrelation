import os
from collections import namedtuple, defaultdict

from inputs import util

DBPEDIA_TRAIN_FILE = "data/dbpedia_csv/train.csv"
DBPEDIA_TEST_FILE = "data/dbpedia_csv/test.csv"
DBPEDIA_VOCAB_FILE = "data/generated/vocab.dbpedia.txt"
DBPEDIA_TRAIN_RECORD = "data/generated/train.dbpedia.tfrecord"
DBPEDIA_TEST_RECORD = "data/generated/test.dbpedia.tfrecord"
DBPEDIA_MAX_LEN = 97

Raw_Example = namedtuple('Raw_Example', 'label entity sentence')

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
      sentence = util.wordpunct_tokenizer( segments[2].strip('"') )

      example = Raw_Example(label, entity, sentence)
      data.append(example)

  return data

def _length_statistics(raw_train_data, raw_test_data):
  '''statistics info of the length'''
  print('DBpedia entity length:')
  length1 = [len(example.entity) for example in raw_train_data]
  length2 = [len(example.entity) for example in raw_test_data]
  util.length_statistics(length1 + length2)

  print('DBpedia sentence length:')
  length1 = [len(example.sentence) for example in raw_train_data]
  length2 = [len(example.sentence) for example in raw_test_data]
  util.length_statistics(length1 + length2)

def load_raw_data(verbose=False):
  train_data = _load_raw_data(DBPEDIA_TRAIN_FILE)
  test_data = _load_raw_data(DBPEDIA_TEST_FILE)
  if verbose:
    _length_statistics(train_data, test_data)
  return train_data, test_data

def build_vocab(raw_data):
  '''collect words in sentence'''
  vocab = set()
  for example in raw_data:
    for w in example.sentence:
        vocab.add(w)
  
  util.write_vocab(vocab, DBPEDIA_VOCAB_FILE)
  return vocab

def _map_tokens_and_pad(raw_example, vocab2id):
  '''inplace map tokens in raw_example to ids
  Args:
    raw_example: type Raw_Example._asdict()
    vocab2id: dict<token, id>
  '''
  entity = util.pad_or_truncate(raw_example['entity'], 3)
  sentence = util.pad_or_truncate(raw_example['sentence'], DBPEDIA_MAX_LEN)
  raw_example['entity'] = util.map_tokens_to_ids(entity, vocab2id)
  raw_example['sentence'] = util.map_tokens_to_ids(sentence, vocab2id)

def _build_sequence_example(raw_example):
  '''build tf.train.SequenceExample from Raw_Example
  context features : label, entity
  sequence features: sentence

  Args: 
    raw_example : type Raw_Example._asdict()

  Returns:
    tf.trian.SequenceExample
  '''
  ex = tf.train.SequenceExample()

  label = raw_example['label']
  ex.context.feature['label'].int64_list.value.append(label)

  entity = raw_example['entity']
  ex.context.feature['entity'].int64_list.value.extend(entity)

  for word_id in raw_example['sentence']:
    word = ex.feature_lists.feature_list['sentence'].feature.add()
    word.int64_list.value.append(word_id)

  return ex
   
def write_as_tfrecord(train_data, test_data, vocab2id):
  '''convert the raw data to TFRecord format and write to disk
  '''
  util.write_as_tfrecord(train_data, 
                         vocab2id, 
                         DBPEDIA_TRAIN_RECORD, 
                         _map_tokens_and_pad, 
                         _build_sequence_example)
  util.write_as_tfrecord(test_data, 
                         vocab2id, 
                         DBPEDIA_TEST_RECORD, 
                         _map_tokens_and_pad, 
                         _build_sequence_example)

  util.shuf_and_write(DBPEDIA_TRAIN_RECORD)

def _parse_tfexample(serialized_example):
  '''parse serialized tf.train.SequenceExample to tensors
  context features : label, entity
  sequence features: sentence
  '''
  context_features={'label'  : tf.FixedLenFeature([], tf.int64),
                    'entity' : tf.FixedLenFeature([3], tf.int64)}
  sequence_features={'sentence' : tf.FixedLenSequenceFeature([], tf.int64)}
  context_dict, sequence_dict = tf.parse_single_sequence_example(
                      serialized_example,
                      context_features   = context_features,
                      sequence_features  = sequence_features)

  sentence = sequence_dict['sentence']
  entity = context_dict['entity']
  label = context_dict['label']

  return label, entity, sentence

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

