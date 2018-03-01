import os
import random
import six
from collections import defaultdict

import numpy as np
import tensorflow as tf

class Vocab(object):
  def __init__(self, data_dir=None, vocab_file=None, vocab_freq_file=None):

    self.pad_token = '<PAD>'
    self.pad_id = 0
    self.unk_token = '<UNK>'
    self.unk_id = 1

    self._vocab = None
    self._vocab2id = None

    
    if vocab_file:
      self.vocab_file = os.path.join(data_dir, vocab_file)
    else:
      self.vocab_file = None
    
    if vocab_freq_file:
      self.vocab_freq_file = os.path.join(data_dir, vocab_freq_file)
    else:
      self.vocab_freq_file = None
    
  def load_vocab(self, vocab_file):
    vocab = []
    with open(vocab_file) as f:
      for line in f:
        token = line.strip()
        vocab.append(token)
    return vocab
  
  def build_vocab2id_dict(self, vocab):
    vocab2id = {}
    for id, token in enumerate(vocab):
      vocab2id[token] = id
    return vocab2id
  
  @property
  def vocab(self):
    if self._vocab is None:
      self._vocab = self.load_vocab(self.vocab_file)
    return self._vocab

  @property
  def vocab2id(self):
    if self._vocab2id is None:
      self._vocab2id = self.build_vocab2id_dict(self.vocab)
    return self._vocab2id

  def encode(self, tokens):
    '''convert a list of tokens to a list of ids
    Args:
      tokens: list, [token0, token1, .. ]
    '''
    ids = []
    for token in tokens:
      if token in self.vocab2id:
        tok_id = self.vocab2id[token]
        ids.append(tok_id)
      elif self.unk_token is not None:
        ids.append(self.unk_id)

    return ids

  def decode(self, ids):
    tokens = []
    for id in ids:
      tok = self.vocab[id]
      tokens.append(tok)
    return tokens
  
  def _count_tokens(self, token_generator, max_vocab_size, min_vocab_freq):
    vocab_freqs = defaultdict(int)

    for token in token_generator:
      vocab_freqs[token] += 1
  
    # Filter out low-occurring terms
    if min_vocab_freq is not None:
      vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items()
                        if vocab_freqs[term] > min_vocab_freq)
    else:
      vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items())

    # Sort by frequency
    ordered_vocab_freqs = sorted(
        vocab_freqs.items(), key=lambda item: item[1], reverse=True)

    # Limit vocab size
    if max_vocab_size is not None:
      ordered_vocab_freqs = ordered_vocab_freqs[:max_vocab_size]

    return ordered_vocab_freqs

  def save(self, file, tokens):
    with open(file, 'w') as f:
      for tok in tokens:
        f.write('%s\n' % tok)
  
  def generate_vocab(self, token_generator, 
                     max_vocab_size=None, min_vocab_freq=None):
    tf.logging.info('generate vocab to %s' % self.vocab_file)
    ordered_vocab_freqs = self._count_tokens(token_generator, 
                                  max_vocab_size, min_vocab_freq)

    tokens = [self.pad_token, self.unk_token]
    tokens.extend([tok for tok, _ in ordered_vocab_freqs])
    self._vocab = tokens

    if self.vocab_file is not None:
      self.save(self.vocab_file, self._vocab)
    
    

    freqs = [10**6, 10]
    freqs.extend([freq for _, freq in ordered_vocab_freqs])
    if self.vocab_freq_file is not None:
      self.save(self.vocab_freq_file, freqs)

  def union(self, other_vocab):
    for tok in other_vocab.vocab:
      if tok not in self.vocab2id:
        self._vocab.append(tok)
    
    self._vocab2id = self.build_vocab2id_dict(self.vocab)
    
    if self.vocab_file is not None:
      self.save(self.vocab_file, self._vocab)

class Embed(object):
  def __init__(self, embed_dir, embed_file, vocab_file):
    self.embed_file = os.path.join(embed_dir, embed_file)
    self.vocab = Vocab(embed_dir, vocab_file)
  
  def load_embedding(self):
    if self.embed_file.endswith('.npz'):
      tensor_dict = np.load(self.embed_file)
      return tensor_dict['word_embed']

    return np.load(self.embed_file) # .npy file
    
  def trim_pretrain_embedding(self, src_embed_obj):
    '''trim unnecessary words from original pre-trained word embedding'''
    tf.logging.info('trim embedding to %s'%self.embed_file)

    src_embed = src_embed_obj.load_embedding()
    src_words2id = src_embed_obj.vocab.vocab2id

    word_embed=[]
    word_dim = src_embed.shape[1]

    for w in self.vocab.vocab:
      if w in src_words2id:
        id = src_words2id[w]
        word_embed.append(src_embed[id])
      else:
        vec = np.random.normal(0,0.1,[word_dim])
        word_embed.append(vec)
    word_embed[self.vocab.pad_id] = np.zeros([word_dim])

    word_embed = np.asarray(word_embed)
    np.save(self.embed_file, word_embed.astype(np.float32))

class Dataset(object):

  def get_length(self):
    raise NotImplementedError

  def length_statistics(self):
    '''get maximum, mean, quantile info for length'''
    length = sorted(self.get_length())
    length = np.asarray(length)

    # p7 = np.percentile(length, 70)
    # Probability{length < p7} = 0.7
    percent = [50, 70, 80, 90, 95, 98, 99, 100]
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

  def set_vocab(self, vocab):
    self.vocab = vocab

  def tokens(self):
    for file in [self.train_file, self.test_file]:
      if file is not None:
        for token in self.token_generator(file):
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

  def __init__(self, out_dir, train_record_file=None, 
               test_record_file=None, unsup_record_file=None):
    self.train_record_file = train_record_file
    if self.train_record_file:
      self.train_record_file = os.path.join(out_dir, self.train_record_file)
      
    self.test_record_file = test_record_file
    if self.test_record_file:
      self.test_record_file = os.path.join(out_dir, self.test_record_file)

    self.unsup_record_file = unsup_record_file
    if self.unsup_record_file:
      self.unsup_record_file = os.path.join(out_dir, self.unsup_record_file)
  
  def _write_records(self, generators, file, shuffle=False):
    writer = tf.python_io.TFRecordWriter(file)
    for generator in generators:
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

  def generate_train_records(self, generators):
    if self.train_record_file:
      self._write_records(generators, self.train_record_file, shuffle=True)

  def generate_test_records(self, generators):
    if self.test_record_file:
      self._write_records(generators, self.test_record_file)

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
