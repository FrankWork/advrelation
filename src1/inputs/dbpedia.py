import os
import tensorflow as tf
from collections import namedtuple, defaultdict

from inputs import util

DBPEDIA_TRAIN_FILE = "data/dbpedia_csv/train.csv"
DBPEDIA_TEST_FILE = "data/dbpedia_csv/test.csv"
DBPEDIA_VOCAB_FILE = "data/generated/vocab.dbpedia.txt"
DBPEDIA_TRAIN_RECORD = "data/generated/train.dbpedia.tfrecord"
DBPEDIA_TEST_RECORD = "data/generated/test.dbpedia.tfrecord"
DBPEDIA_MAX_LEN = 97

Raw_Example = namedtuple('Raw_Example', 'label entity pos sentence')

def _find_entity_pos(entity, sentence):
  pos = []
  for e_tok in entity:
    for i, s_tok in enumerate(sentence):
      if e_tok == s_tok:
        pos.append(i)
        if len(pos) == 2:
          break
  n = len(pos)
  if n==0:
    pos.append(0)
    pos.append(len(sentence))
  elif n==1:
    pos *= 2

  return pos

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
      segments = [x.strip('"') for x in segments]
      assert len(segments) == 3
      
      label = int(segments[0])
      label += 19
      entity   = util.split_by_punct( segments[1])
      sentence = util.wordpunct_tokenizer( segments[2])

      pos = _find_entity_pos(entity, sentence)

      example = Raw_Example(label, entity, pos, sentence)
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
    for w in example.entity:
      vocab.add(w)
  
  util.write_vocab(vocab, DBPEDIA_VOCAB_FILE)
  return vocab

def _map_tokens_and_pad(raw_example, vocab2id):
  '''inplace map tokens in raw_example to ids
  Args:
    raw_example: type Raw_Example._asdict()
    vocab2id: dict<token, id>
  '''
  raw_example['entity'] = util.map_tokens_to_ids(raw_example['entity'], vocab2id)

  sentence = util.pad_or_truncate(raw_example['sentence'], DBPEDIA_MAX_LEN)
  raw_example['sentence'] = util.map_tokens_to_ids(sentence, vocab2id)

def _lexical_feature(raw_example):
  def _entity_context(e_idx, sent):
    ''' return [w(e-1), w(e), w(e+1)]
    '''
    context = []
    if e_idx >= len(sent):
      e_idx = len(sent)-1
    
    context.append(sent[e_idx])

    if e_idx >= 1:
      context.append(sent[e_idx-1])
    else:
      context.append(sent[e_idx])
    
    if e_idx < len(sent)-1:
      context.append(sent[e_idx+1])
    else:
      context.append(sent[e_idx])
    
    return context

  entity = raw_example['entity']
  if len(entity) > 3:
    entity = util.pad_or_truncate(entity, 3)
  elif len(entity) < 3:
    e_idx = raw_example['pos'][0]
    entity = _entity_context(e_idx, raw_example['sentence'])

  e_idx = raw_example['pos'][1]
  context = _entity_context(e_idx, raw_example['sentence'])
  
  lexical = entity + context
  return lexical

def _position_feature(raw_example):
  e1_idx = raw_example['pos'][0]
  e2_idx = raw_example['pos'][1]

  position1 = []
  position2 = []
  length = len(raw_example['sentence'])
  for i in range(length):
    position1.append(util.relative_distance(i-e1_idx))
    position2.append(util.relative_distance(i-e2_idx))
  
  return position1, position2

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

  lexical = _lexical_feature(raw_example)
  ex.context.feature['lexical'].int64_list.value.extend(lexical)

  label = raw_example['label']
  ex.context.feature['label'].int64_list.value.append(label)

  for word_id in raw_example['sentence']:
    word = ex.feature_lists.feature_list['sentence'].feature.add()
    word.int64_list.value.append(word_id)
  
  position1, position2 = _position_feature(raw_example)
  for pos_val in position1:
    pos = ex.feature_lists.feature_list['position1'].feature.add()
    pos.int64_list.value.append(pos_val)
  for pos_val in position2:
    pos = ex.feature_lists.feature_list['position2'].feature.add()
    pos.int64_list.value.append(pos_val)

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
  context features : lexical, label
  sequence features: sentence, position1, position2
  '''
  context_features={
                      'lexical'   : tf.FixedLenFeature([6], tf.int64),
                      'label'    : tf.FixedLenFeature([], tf.int64)}
  sequence_features={
                      'sentence' : tf.FixedLenSequenceFeature([], tf.int64),
                      'position1'  : tf.FixedLenSequenceFeature([], tf.int64),
                      'position2'  : tf.FixedLenSequenceFeature([], tf.int64)}
  context_dict, sequence_dict = tf.parse_single_sequence_example(
                      serialized_example,
                      context_features   = context_features,
                      sequence_features  = sequence_features)

  sentence = sequence_dict['sentence']
  position1 = sequence_dict['position1']
  position2 = sequence_dict['position2']

  lexical = context_dict['lexical']
  label = context_dict['label']

  return lexical, label, sentence, position1, position2

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

