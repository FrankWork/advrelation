import re
import os
import random

import numpy as np
import tensorflow as tf
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import generator_utils

SEMEVAL_DATA_DIR = "data/SemEval"
TRAIN_FILE = os.path.join(SEMEVAL_DATA_DIR, "train.txt")
TEST_FILE = os.path.join(SEMEVAL_DATA_DIR, "test.txt")
LABEL_FILE = os.path.join(SEMEVAL_DATA_DIR, "relations.txt")

OUTPUT_DIR = "data/generated"
TRAIN_RECORD = os.path.join(OUTPUT_DIR, "train.tfrecord")
TEST_RECORD = os.path.join(OUTPUT_DIR, "test.tfrecord")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.txt")

VOCAB_SIZE = 2**13 # 8k, 22k
VOCAB_FILE = "semeval.vocab"

class RawDataGenerator(object):
  '''Load text from file'''
  def __init__(self):
    self.entity_finder = re.compile(r"<e[12]>(.*?)</e[12]>")
    self.entity_tag_mask = re.compile(r"</?e[12]>")
    self.space_mask = re.compile(r'\s{2,}')

  def find_start_position(self, entities, sentence):
    ''' find start position of the entity in sentence
    Args:
      entities: a list of 2 entities, each entity is a list of tokens
      sentence: a list of tokens
    '''
    pos = []
    for entity in entities:
      n = len(entity)
      for i in range(len(sentence)):
        if sentence[i:i+n]==entity:
          # first, last = i, i+n-1
          pos.append(i)
    return pos

  def relative_distance(self, n):
    '''convert relative distance to positive number
    -60), [-60, 60], (60
    '''
    if n < -60:
      return 0
    elif n >= -60 and n <= 60:
      return n + 61
    
    return 122

  def position_feature(self, ent_pos, sentence):
    '''
    Args:
      ent_pos: int, start position of the entity
      sentence: a list of tokens
    '''
    length = len(sentence)
    return [self.relative_distance(i-ent_pos) for i in range(length)]
  
  def entity_context(self, ent_pos, sentence):
    ''' return [w(e-1), w(e), w(e+1)]
    '''
    context = []
    context.append(sentence[ent_pos])

    if ent_pos >= 1:
      context.append(sentence[ent_pos-1])
    else:
      context.append(sentence[ent_pos])
    
    if ent_pos < len(sentence)-1:
      context.append(sentence[ent_pos+1])
    else:
      context.append(sentence[ent_pos])
    
    return context 
  
  def lexical_feature(self, entities_pos, sentence):
    context1 = self.entity_context(entities_pos[0], sentence)
    context2 = self.entity_context(entities_pos[1], sentence)

    # ignore WordNet hypernyms in paper
    return context1 + context2

  def generator(self, data_files, for_vocab=False):
    '''load raw data from text file, 
    file contents:
      1	"The ... an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
      Component-Whole(e2,e1)
      Comment: Not a collection: there is structure here, organisation.

      EOF
    Returns:
      label: string
      entities: list of string, len == 2
      sentence: string
    '''
    if not isinstance(data_files, list):
      data_files = [data_files]
    
    for data_file in data_files:
      lines = open(data_file).readlines()
      n = len(lines)
      assert n % 4 == 0
      for i in range(n//4):
        text = lines[4*i].split('\t')[1].strip('"|\n')
        sentence = self.entity_tag_mask.sub(' ', text)
        sentence = self.space_mask.sub(" ", sentence)

        if for_vocab:
          yield sentence
        else:
          entities = self.entity_finder.findall(text)
          assert len(entities) == 2

          label = lines[4*i+1].strip()

          yield label, entities, sentence

class SemEval2010Task8(object):
  def __init__(self):
    self.raw_gen = RawDataGenerator()
  
  def generate_vocab(self):
    # Generate vocab
    token_generator = self.raw_gen.generator([TRAIN_FILE, TEST_FILE], for_vocab=True)
    self.vocab_encoder = generator_utils.get_or_generate_vocab_inner(
                            OUTPUT_DIR, VOCAB_FILE, VOCAB_SIZE,token_generator)
    tf.logging.info('vocab_size', self.vocab_encoder.vocab_size)
    self.label_encoder = text_encoder.ClassLabelEncoder(class_labels_fname=LABEL_FILE)
  
  def generator(self, data_file):
    """Generate examples."""
    for label, entities, sentence in self.raw_gen.generator(data_file):
      entities = [self.vocab_encoder.encode(e) for e in entities]
      sentence = self.vocab_encoder.encode(sentence)

      entities_pos = self.raw_gen.find_start_position(entities, sentence)
      
      yield {
          "sentence": sentence,
          "label": self.label_encoder.encode(label),
          'lexical': self.raw_gen.lexical_feature(entities_pos, sentence),
          'position1': self.raw_gen.position_feature(entities_pos[0], sentence),
          'position2': self.raw_gen.position_feature(entities_pos[1], sentence),
      }

  def generate_data(self):
    train_generator = self.generator(TRAIN_FILE)
    test_generator = self.generator(TEST_FILE)

    self._generate_data(train_generator, TRAIN_RECORD, True)
    self._generate_data(test_generator, TEST_RECORD, False)
  
  def _generate_data(self, generator, file, shuffle=False):
    writer = tf.python_io.TFRecordWriter(file)
    for example in generator:
      tf_example = self.build_tfexample(example)
      writer.write(tf_example.SerializeToString())
    writer.close()

    if shuffle:
      self.shuffle_records(file)
  
  def shuffle_records(self, filename):
    reader = tf.python_io.tf_record_iterator(filename)
    records = []
    for record in reader:
      # record is of <class 'bytes'>
      records.append(record)
    reader.close()

    random.shuffle(records)
    
    writer = tf.python_io.TFRecordWriter(filename)
    for record in records:
      writer.write(record)
    writer.close()
    
  def build_tfexample(self, raw_example):
    '''build tf.train.SequenceExample from dict
    context features : lexical, rid
    sequence features: sentence, position1, position2

    Args: 
      raw_example : type dict

    Returns:
      tf.trian.SequenceExample
    '''
    ex = tf.train.SequenceExample()

    lexical = raw_example['lexical']
    ex.context.feature['lexical'].int64_list.value.extend(lexical)

    label = raw_example['label']
    ex.context.feature['label'].int64_list.value.append(label)

    for word_id in raw_example['sentence']:
      word = ex.feature_lists.feature_list['sentence'].feature.add()
      word.int64_list.value.append(word_id)
    
    for pos_val in raw_example['position1']:
      pos = ex.feature_lists.feature_list['position1'].feature.add()
      pos.int64_list.value.append(pos_val)
    for pos_val in raw_example['position2']:
      pos = ex.feature_lists.feature_list['position2'].feature.add()
      pos.int64_list.value.append(pos_val)

    return ex

  def get_length(self):
    length = []
    for i, file in enumerate([TRAIN_RECORD, TEST_RECORD]):
      reader = tf.python_io.tf_record_iterator(file)
      for record in reader:
        x = tf.train.SequenceExample()
        x.ParseFromString(record)
        n = len(x.feature_lists.feature_list['sentence'].feature)
        length.append(n)
    return length

  def parse_tfexample(self, serialized_example):
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

  def read_data(self, epoch, batch_size):
    train_iter = self.read_tfrecord(TRAIN_RECORD, epoch, batch_size, True)
    test_iter = self.read_tfrecord(TEST_RECORD, epoch, batch_size, False)
    return train_iter, test_iter

  def read_tfrecord(self, filename, epoch, batch_size, shuffle=True):
    '''read TFRecord file to get batch tensors for tensorflow models

    Returns:
      a tuple of batched tensors
    '''
    with tf.device('/cpu:0'):
      dataset = tf.data.TFRecordDataset([filename])
      # Parse the record into tensors
      dataset = dataset.map(self.parse_tfexample)
      dataset = dataset.repeat(epoch)
      if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
      
      padded_shapes = ([None,], [], [None], [None], [None])
      dataset = dataset.padded_batch(batch_size, padded_shapes)
      # dataset = dataset.batch(batch_size)
      
      if shuffle:
        iterator = dataset.make_one_shot_iterator()
      else:
        iterator = dataset.make_initializable_iterator()
      return iterator

_dataset = SemEval2010Task8()

def generate_data():
  _dataset.generate_vocab()
  _dataset.generate_data()
  length = _dataset.get_length()
  length_statistics(length)

def read_data(epoch, batch_size):
  return _dataset.read_data(epoch, batch_size)

def write_results(predictions):
  label_encoder = text_encoder.ClassLabelEncoder(class_labels_fname=LABEL_FILE)
  
  start_no = 8001
  with open(FLAGS.results_file, 'w') as f:
    for idx, id in enumerate(predictions):
      if idx < 2717:
        rel = label_encoder.decode(id)
        f.write('%d\t%s\n' % (start_no+idx, rel))

def length_statistics(length):
  '''get maximum, mean, quantile from length
  Args:
    length: list<int>
  '''
  length = sorted(length)
  length = np.asarray(length)

  # p7 = np.percentile(length, 70)
  # Probability{length < p7} = 0.7
  percent = [50, 70, 80, 90, 95, 98, 100]
  quantile = [np.percentile(length, p) for p in percent]
  
  print('(percent, quantile)', list(zip(percent, quantile)))
