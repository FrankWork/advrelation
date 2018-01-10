import os
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf

PRETRAIN_DIR = "data/pretrain"

GOOGLE_EMBED300_FILE = "embed300.google.npy"
GOOGLE_WORDS_FILE = "google_words.lst"
TRIMMED_EMBED300_FILE = "embed300.trim.npy

class VocabMgr(object):
  def __init__(self, out_dir, vocab_file, vocab_freq_file=None,
                     max_vocab_size=None, min_vocab_freq=None):
    self.out_dir = out_dir
    self.vocab_file = os.path.join(out_dir, vocab_file)
    if vocab_freq_file:
      self.vocab_freq_file = os.path.join(out_dir, vocab_freq_file)
    else:
      self.vocab_freq_file = None
    self.max_vocab_size = max_vocab_size
    self.min_vocab_freq = min_vocab_freq
    self.pad_token = '<PAD>'
    self.pad_id = 0

    self._vocab = None
    self._vocab2id = None
  
  def _load_vocab_from_file(self, vocab_file):
    vocab = []
    with open(vocab_file) as f:
      for line in f:
        w = line.strip()
        vocab.append(w)
    return vocab

  def _build_vocab2id(self, vocab):
    vocab2id = {}
    for id, token in enumerate(vocab):
      vocab2id[token] = id
    return vocab2id

  @property
  def vocab(self):
    if self._vocab is None:
      self._vocab = self._load_vocab_from_file(self.vocab_file)
    return self._vocab

  @property
  def vocab2id(self):
    if self._vocab2id is None:
      self._vocab2id = self._build_vocab2id(self, self.vocab)
    return self._vocab2id

  def _generate_vocab_inner(self, token_generator):
    vocab_freqs = defaultdict(int)

    for token in token_generator:
      vocab_freqs[token] += 1
  
    # Filter out low-occurring terms
    if self.min_vocab_freq is not None:
      vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items()
                        if vocab_freqs[term] > self.min_vocab_freq)
    else:
      vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items())

    # Sort by frequency
    ordered_vocab_freqs = sorted(
        vocab_freqs.items(), key=lambda item: item[1], reverse=True)

    # Limit vocab size
    if self.max_vocab_size is not None:
      ordered_vocab_freqs = ordered_vocab_freqs[:self.max_vocab_size]

    return ordered_vocab_freqs

  def generate_vocab(self, token_generator):
    tf.logging.info('generate vocab to %s' % self.vocab_file)
    
    ordered_vocab_freqs = self._generate_vocab_inner(token_generator)

    freq_f = None
    if self.vocab_freq_file:
      freq_f = open(self.vocab_freq_file, 'w')

    with open(self.vocab_file, 'w') as vocab_f:
      vocab_f.write('%s\n'% self.pad_token)
      if freq_f:
        freq_f.write('%d\n' % 10**6)
      for token, freq in ordered_vocab_freqs:
        vocab_f.write('%s\n'% token)
        if freq_f:
          freq_f.write('%d\n' % freq)
    
    if freq_f:
      freq_f.close()

  def trim_pretrain_embedding(self, trimed_embed_file=TRIMMED_EMBED300_FILE, 
                              pretrain_dir=PRETRAIN_DIR, 
                              pretrain_embed_file=GOOGLE_EMBED300_FILE, 
                              pretrain_vocab_file=GOOGLE_WORDS_FILE):
    '''trim unnecessary words from original pre-trained word embedding'''
    pretrain_embed_file = os.path.join(pretrain_dir, pretrain_embed_file)
    pretrain_vocab_file = os.path.join(pretrain_dir, pretrain_vocab_file)
    trimed_embed_file = os.path.join(self.out_dir, trimed_embed_file)

    tf.logging.info('trim embedding to %s'%trimed_embed_file)

    pretrain_embed    = np.load(pretrain_embed_file)
    pretrain_vocab = self._load_vocab_from_file(pretrain_vocab_file)
    pretrain_words2id = self._build_vocab2id(pretrain_vocab)

    word_embed=[]
    word_dim = pretrain_embed.shape[1]

    for w in self.vocab:
      if w in pretrain_words2id:
        id = pretrain_words2id[w]
        word_embed.append(pretrain_embed[id])
      else:
        vec = np.random.normal(0,0.1,[word_dim])
        word_embed.append(vec)
    word_embed[self.pad_id] = np.zeros([word_dim])

    word_embed = np.asarray(word_embed)
    np.save(trimed_embed_file, word_embed.astype(np.float32))

  def load_embedding(self, embed_file=TRIMMED_EMBED300_FILE):
    return np.load(embed_file)

  def map_token_to_id(self, tokens):
    '''convert a list of tokens to a list of ids
    Args:
      tokens: list, [token0, token1, .. ]
    '''
    ids = []
    for token in tokens:
      if token in self.token2id:
        tok_id = self.token2id[token]
        ids.append(tok_id)
    return ids

class Dataset(object):

  def get_length(self):
    raise NotImplementedError

  def length_statistics(self):
    '''get maximum, mean, quantile info for length'''
    length = sorted(self.get_length())
    length = np.asarray(length)

    # p7 = np.percentile(length, 70)
    # Probability{length < p7} = 0.7
    percent = [50, 70, 80, 90, 95, 98, 100]
    quantile = [np.percentile(length, p) for p in percent]
    
    tf.loggin.info('(percent, quantile)', list(zip(percent, quantile)))

class TextDataset(Dataset):
  def __init__(self, data_dir, train_file, test_file, vocab_mgr):
    self.train_file = os.path.join(self.data_dir, train_file)
    self.test_file = os.path.join(self.data_dir, test_file)
    self.vocab_mgr = vocab_mgr

  def generate_vocab(self):
    self.vocab_mgr.generate_vocab(self.tokens())
    self.vocab_mgr.trim_pretrain_embedding()

  def tokens(self):
    for token in self.token_generator(self.train_file):
      yield token

    for token in self.token_generator(self.test_file):
      yield token

  def token_generator(self, file):
    raise NotImplementedError
  
  def example_generator(self, file):
    raise NotImplementedError
  
  def train_examples(self):
    for example in self.example_generator(self.train_file):
      yield example

  def test_examples(self):
    for example in self.example_generator(self.test_file):
      yield example

class RecordDataset(Dataset):
  def __init__(self, out_dir, train_record_file, test_record_file,
               text_dataset):
    self.out_dir = out_dir
    self.train_record_file = os.path.join(self.out_dir, train_record_file)
    self.test_record_file = os.path.join(self.out_dir, test_record_file)
    self.text_dataset = text_dataset
  
  def example_generator(self, raw_example_generator):
    """Generate examples."""
    raise NotImplementedError

  def generate_data(self):
    def _generate_data(generator, file, shuffle=False):
      writer = tf.python_io.TFRecordWriter(file)
      for example in generator:
        tf_example = self.build_tfexample(example)
        writer.write(tf_example.SerializeToString())
      writer.close()
      if shuffle:
        self.shuffle_records(file)

    train_generator = self.generator(TRAIN_FILE)
    test_generator = self.generator(TEST_FILE)

    self._generate_data(train_generator, TRAIN_RECORD, True)
    self._generate_data(test_generator, TEST_RECORD, False)
    
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
