import re
import os
import tensorflow as tf
from collections import namedtuple, defaultdict

from inputs import util

flags = tf.app.flags

flags.DEFINE_string("semeval_train_file", "data/SemEval/train.txt", 
                             "original training file")
flags.DEFINE_string("semeval_test_file", "data/SemEval/test.txt", 
                             "original test file")

flags.DEFINE_string("semeval_vocab_file", "data/generated/vocab.semeval.txt", 
                              "vocab of train and test data")

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

flags.DEFINE_integer("semeval_max_len", 97, "max length of sentences")

FLAGS = flags.FLAGS


Raw_Example = namedtuple('Raw_Example', 'label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair', 'first last')

_entity_regex = re.compile(r"<e[12]>(.*?)</e[12]>")
_etag_mask = re.compile(r"</?e[12]>")

def _load_relations():
  id2relation = []
  relation2id = dict()

  with open(FLAGS.relations_file) as f:
    for id, line in enumerate(f):
      rel = segment = line.strip().split()[1]
      id2relation.append(rel)
      relation2id[rel] = id
  
  return relation2id, id2relation

def _find_entity_pos(entity, tokens):
  ''' find start and ending position of the entity in tokens
  '''
  n = len(entity)
  for i in range(len(tokens)):
    if tokens[i:i+n]==entity:
      first, last = i, i+n-1
      return PositionPair(first, last)

def _load_raw_data(filename, relation2id):
  '''load raw data from text file, 
  file contents:
    1	"The ... an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
    Component-Whole(e2,e1)
    Comment: Not a collection: there is structure here, organisation.

    2	"The <e1>child</e1> ... the <e2>cradle</e2> by means of a cord."
    Other
    Comment:

    EOF

  return: a list of Raw_Example
  '''
  data = []
  lines = open(filename).readlines()
  n = len(lines)
  assert n % 4 == 0
  for i in range(n//4):
    sentence = lines[4*i].split('\t')[1].strip('"|\n').lower()
    
    entities = _entity_regex.findall(sentence)
    assert len(entities) == 2

    sentence = _etag_mask.sub(' ', sentence)
    tokens = util.wordpunct_tokenizer(sentence)

    entities = [util.wordpunct_tokenizer(entity) for entity in entities]
    entity1 = _find_entity_pos(entities[0], tokens)
    entity2 = _find_entity_pos(entities[1], tokens)
    assert entity1 is not None and entity2 is not None

    rel_text = lines[4*i+1].strip()
    label = relation2id[rel_text]

    example = Raw_Example(label, entity1, entity2, tokens)
    data.append(example)

  return data

def _length_statistics(raw_train_data, raw_test_data):
  '''statistics info of the length'''
  print('SemEval sentence length:')
  length1 = [len(example.sentence) for example in raw_train_data]
  length2 = [len(example.sentence) for example in raw_test_data]
  util.length_statistics(length1 + length2)

def load_raw_data(verbose=False):
  relation2id, _ = _load_relations()
  train_data = _load_raw_data(FLAGS.semeval_train_file, relation2id)
  test_data = _load_raw_data(FLAGS.semeval_test_file, relation2id)
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

  vocab = [token for token, _ in ordered_vocab_freqs]
  util.write_vocab(vocab, FLAGS.semeval_vocab_file)
  return vocab

def _map_tokens_and_pad(raw_example, vocab2id):
  '''inplace map tokens in raw_example to ids
  Args:
    raw_example: type Raw_Example._asdict()
    vocab2id: dict<token, id>
  '''
  sentence = util.map_tokens_to_ids(raw_example['sentence'], vocab2id)
  raw_example['sentence'] = util.pad_or_truncate(sentence, FLAGS.semeval_max_len)

def _lexical_feature(raw_example):
  def _entity_context(e_idx, sent):
    ''' return [w(e-1), w(e), w(e+1)]
    '''
    context = []
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

    
  e1_idx = raw_example['entity1'].first
  e2_idx = raw_example['entity2'].first

  context1 = _entity_context(e1_idx, raw_example['sentence'])
  context2 = _entity_context(e2_idx, raw_example['sentence'])

  # ignore WordNet hypernyms in paper
  lexical = context1 + context2
  return lexical

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
  train_data = util.read_tfrecord(FLAGS.semeval_train_record, 
                              epoch, 
                              batch_size, 
                              _parse_tfexample,
                              shuffle=True)
  test_data = util.read_tfrecord(FLAGS.semeval_test_record, 
                              epoch, 
                              batch_size, 
                              _parse_tfexample,
                              shuffle=False)

  return train_data, test_data

def write_results(predictions):
  _, id2relations = _load_relations()
  
  start_no = 8001
  with open(FLAGS.results_file, 'w') as f:
    for idx, id in enumerate(predictions):
      rel = id2relations[id]
      f.write('%d\t%s\n' % (start_no+idx, rel))
