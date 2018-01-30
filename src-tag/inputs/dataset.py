import os
import random
import six
from collections import defaultdict

import numpy as np
import tensorflow as tf

PRETRAIN_DIR = "data/pretrain"
OUT_DIR = "data/generated"

GOOGLE_EMBED300_FILE = "embed300.google.npy"
GOOGLE_WORDS_FILE = "google_words.lst"
TRIMMED_EMBED300_FILE = "embed300.trim.npy"

VOCAB_SIZE = None#2**13 # 8k, 22k
VOCAB_FILE = "vocab.txt"

class VocabMgr(object):
  def __init__(self, out_dir=OUT_DIR, vocab_file=VOCAB_FILE, 
                vocab_freq_file=None, max_vocab_size=None, min_vocab_freq=None):
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
      self._vocab2id = self._build_vocab2id(self.vocab)
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
    return np.load(os.path.join(OUT_DIR, embed_file))

  def map_token_to_id(self, tokens):
    '''convert a list of tokens to a list of ids
    Args:
      tokens: list, [token0, token1, .. ]
    '''
    ids = []
    for token in tokens:
      if token in self.vocab2id:
        tok_id = self.vocab2id[token]
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
    
    tf.logging.info('(percent, quantile) %s' % str(list(zip(percent, quantile))))

class TextDataset(Dataset):
  def __init__(self, data_dir, max_len=None, train_file=None, test_file=None, unsup_file=None):
    self.max_len = max_len
    self.train_file = train_file
    if self.train_file:
      self.train_file = os.path.join(data_dir, self.train_file)
      
    self.test_file = test_file
    if self.test_file:
      self.test_file = os.path.join(data_dir, self.test_file)

    self.unsup_file = unsup_file
    if self.unsup_file:
      self.unsup_file = os.path.join(data_dir, self.unsup_file)

  def set_vocab_mgr(self, vocab_mgr):
    self.vocab_mgr = vocab_mgr

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
    if self.train_examples:
      for example in self.example_generator(self.train_file):
        yield example

  def test_examples(self):
    if self.test_file:
      for example in self.example_generator(self.test_file):
        yield example
  
  def unsup_examples(self):
    if self.unsup_file:
      for example in self.example_generator(self.unsup_file):
        yield example

class RecordDataset(Dataset):

  def __init__(self, text_dataset, out_dir=OUT_DIR, train_record_file=None, 
               test_record_file=None, unsup_record_file=None):
    self.text_dataset = text_dataset

    self.train_record_file = train_record_file
    if self.train_record_file:
      self.train_record_file = os.path.join(out_dir, self.train_record_file)
      
    self.test_record_file = test_record_file
    if self.test_record_file:
      self.test_record_file = os.path.join(out_dir, self.test_record_file)

    self.unsup_record_file = unsup_record_file
    if self.unsup_record_file:
      self.unsup_record_file = os.path.join(out_dir, self.unsup_record_file)
  
  def _write_records(self, generator, file, shuffle=False):
    writer = tf.python_io.TFRecordWriter(file)
    for example in generator:
      example = to_example(example)
      writer.write(example.SerializeToString())
    writer.close()
    if shuffle:
      self._shuffle_records(file)

  def _shuffle_records(self, filename):
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

  def generate_data(self):
    tf.logging.info('generate TFRecord data')
    if self.train_record_file:
      train_gen = self.text_dataset.train_examples()
      self._write_records(train_gen, self.train_record_file, shuffle=True)

    if self.test_record_file:
      test_gen = self.text_dataset.test_examples()
      self._write_records(test_gen, self.test_record_file)

    if self.unsup_record_file:
      unsup_gen = self.text_dataset.unsup_examples()
      self._write_records(unsup_gen, self.unsup_record_file, shuffle=True)

  def get_length(self):
    length = []
    for i, file in enumerate([TRAIN_RECORD, TEST_RECORD]):
      reader = tf.python_io.tf_record_iterator(file)
      for record in reader:
        x = tf.train.SequenceExample()
        x.ParseFromString(record)
        n = len(x.features.feature["sentence"].int64_list.value)
        length.append(n)
    return length

  def parse_example(self, example):
    raise NotImplementedError
  
  def padded_shapes(self):
    raise NotImplementedError

  def train_data(self, epoch, batch_size):
    if self.train_record_file:
      return self._read_records(self.train_record_file, epoch, batch_size, 
                                shuffle=True)

  def test_data(self, epoch, batch_size):
    if self.test_record_file:
      return self._read_records(self.test_record_file, 1, batch_size, 
                                shuffle=False)
  
  def unsup_data(self, epoch, batch_size):
    if self.unsup_record_file:
      return self._read_records(self.unsup_record_file, epoch, batch_size, 
                                shuffle=True)

  def _read_records(self, filename, epoch, batch_size, shuffle=True):
    '''read TFRecord file to get batch tensors for tensorflow models

    Returns:
      a tuple of batched tensors
    '''
    with tf.device('/cpu:0'):
      dataset = tf.data.TFRecordDataset([filename])
      # Parse the record into tensors
      dataset = dataset.map(self.parse_example)
      dataset = dataset.repeat(epoch)
      if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
      
      dataset = dataset.padded_batch(batch_size, self.padded_shapes())
      
      if shuffle:
        iterator = dataset.make_one_shot_iterator()
      else:
        iterator = dataset.make_initializable_iterator()
      return iterator

def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s", str((k, v)))
    if isinstance(v[0], six.integer_types):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))
