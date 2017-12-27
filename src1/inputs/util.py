import os
import re
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from collections import namedtuple

flags = tf.app.flags

flags.DEFINE_string("vocab_file", "data/generated/vocab.txt", 
                              "vocab of train and test data")

flags.DEFINE_string("google_embed300_file", 
                             "data/pretrain/embed300.google.npy", 
                             "google news word embeddding")
flags.DEFINE_string("google_words_file", 
                             "data/pretrain/google_words.lst", 
                             "google words list")
flags.DEFINE_string("trimmed_embed300_file", 
                             "data/generated/embed300.trim.npy", 
                             "trimmed google embedding")

flags.DEFINE_string("senna_embed50_file", 
                             "data/pretrain/embed50.senna.npy", 
                             "senna words embeddding")
flags.DEFINE_string("senna_words_file", 
                             "data/pretrain/senna_words.lst", 
                             "senna words list")
flags.DEFINE_string("trimmed_embed50_file", 
                             "data/generated/embed50.trim.npy", 
                             "trimmed senna embedding")
FLAGS = tf.app.flags.FLAGS # load FLAGS.word_dim

PAD_WORD = "<pad>"

# similar to nltk.tokenize.regexp.WordPunctTokenizer
# decimal, inter, 'm, 's, 'll, 've, 're, 'd, n't, words, punctuations
regexp = re.compile(r"\d*\.\d+|\d+|'m|'s|'ll|'ve|'re|'d|n't|\w+|[^\w\s]+")

def wordpunct_tokenizer(line):
  '''tokenizer sentence by decimal, inter, 
  'm, 's, 'll, 've, 're, 'd, n't, words, punctuations
  '''
  # replace html tags, <br /> in imdb text
  line = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", line)
  line = re.sub(r'<[^>]*>', ' ', line)
  line = re.sub(r"n't", " n't", line)
  return regexp.findall(line)

def split_by_punct(segment):
  """Splits str segment by punctuation, filters our empties and spaces."""
  return [s for s in re.split(r'\W+', segment) if s and not s.isspace()]

def write_vocab(vocab, vocab_file=FLAGS.vocab_file):
  '''write vocab to the file
  
  Args:
    vocab: a list of tokens
    vocab_file: filename of the file
  '''
  with open(vocab_file, 'w') as f:
    f.write('%s\n' % PAD_WORD) # make sure the pad id is 0
    for w in vocab:
      f.write('%s\n' % w)

def _load_vocab(vocab_file):
  # load vocab from file
  vocab = []
  with open(vocab_file) as f:
    for line in f:
      w = line.strip()
      vocab.append(w)

  return vocab

def load_embedding(embed_file=None, word_dim=None):
  '''Load embeddings from file
  '''
  if embed_file is None:
    if word_dim == 50:
      embed_file   = FLAGS.trimmed_embed50_file
    elif word_dim == 300:
      embed_file   = FLAGS.trimmed_embed300_file
  
  embed = np.load(embed_file)
  return embed 

def load_vocab2id(vocab_file=None):
  if vocab_file is None:
    vocab_file = FLAGS.vocab_file
  
  vocab2id = {}
  vocab = _load_vocab(vocab_file)
  for id, token in enumerate(vocab):
    vocab2id[token] = id

  return vocab2id

def trim_embeddings(word_dim):
  '''trim unnecessary words from original pre-trained word embedding'''
  print('word_dim %d'%word_dim)
  if word_dim == 50:
    pretrain_embed_file = FLAGS.senna_embed50_file
    pretrain_words_file = FLAGS.senna_words_file
    trimed_embed_file   = FLAGS.trimmed_embed50_file
  elif word_dim == 300:
    pretrain_embed_file = FLAGS.google_embed300_file
    pretrain_words_file = FLAGS.google_words_file
    trimed_embed_file   = FLAGS.trimmed_embed300_file

  pretrain_embed    = load_embedding(pretrain_embed_file, word_dim)
  pretrain_words2id = load_vocab2id(pretrain_words_file)

  word_embed=[]
  vocab = _load_vocab(FLAGS.vocab_file)
  for w in vocab:
    if w in pretrain_words2id:
      id = pretrain_words2id[w]
      word_embed.append(pretrain_embed[id])
    else:
      vec = np.random.normal(0,0.1,[word_dim])
      word_embed.append(vec)
  pad_id = -1
  word_embed[pad_id] = np.zeros([word_dim])

  word_embed = np.asarray(word_embed)
  np.save(trimed_embed_file, word_embed.astype(np.float32))

def length_statistics(length):
  '''get maximum, mean, quantile from length
  Args:
    length: list<int>
  '''
  length = sorted(length)
  length = np.asarray(length)

  max_len = np.max(length)
  avg_len = np.mean(length)

  # p7 = np.percentile(length, 70)
  # Probability{length < p7} = 0.7
  percent = [50, 70, 80, 90, 95, 98, 100]
  quantile = [np.percentile(length, p) for p in percent]
  
  print('(percent, quantile)', list(zip(percent, quantile)))

def map_tokens_to_ids(tokens, token2id):
  '''convert a list of tokens to a list of ids
  Args:
    tokens: list, [token0, token1, .. ]
    token2id: dict<token, id> {token0: id0, ...}
  '''
  ids = []
  for token in tokens:
    if token in token2id:
      tok_id = token2id[token]
      ids.append(tok_id)
  return ids

def relative_distance(n):
  '''convert relative distance to positive number
  -60), [-60, 60], (60
  '''
  if n < -60:
    return 0
  elif n >= -60 and n <= 60:
    return n + 61
  
  return 122

def pad_or_truncate(tokens, max_len, pad_val=0):
  '''pad or truncate a list of tokens to max_len
  Args:
    tokens: list, [token0, token1, .. ]
    max_len: int, truncate if len(tokens) > max_len
    pad_val: padding value, pad if len(tokens) < max_len
  '''
  # truncate if len(sentence) > max_len
  # else nothing happens
  tokens = tokens[:max_len] 

  # pad if len(sentence) < max_len
  # else nothing happens
  pad_n = max_len - len(tokens)
  tokens.extend(pad_n*[pad_val])
  
  return tokens

def _write_text_for_debug(text_writer, raw_example, vocab2id):
  '''write raw_example['sentence'] to the disk, for debug 

  Args:
    text_writer: text_writer = open(file, 'w')
    raw_example: an instance of Raw_Example._asdict()
    vocab2id: dict<token, id> {token0: id0, ...}
  '''
  tokens = []
  for token in raw_example['sentence']:
    if token in vocab2id:
      tokens.append(token)
  text_writer.write(' '.join(tokens) + '\n')
      
def write_as_tfrecord(raw_data, vocab2id, filename, map_func, build_func):
  '''convert the raw data to TFRecord format and write to disk

  Args:
    raw_data: a list of Raw_Example
    vocab2id: dict<token, id>
    filename: file to write in
    map_func: function to inplace convert Raw_Example from token to ids
    build_func: function to convert Raw_Example to tf.train.SequenceExample
  '''
  writer = tf.python_io.TFRecordWriter(filename)
  # text_writer = open(filename+'.debug.txt', 'w')
  pad_id = vocab2id[PAD_WORD]
  
  for raw_example in raw_data:
    raw_example = raw_example._asdict()

    # _write_text_for_debug(text_writer, raw_example, vocab2id)

    map_func(raw_example, vocab2id)
    example = build_func(raw_example)
    writer.write(example.SerializeToString())
  writer.close()
  # text_writer.close()
  del raw_data

def read_tfrecord(filename, epoch, batch_size, parse_func, shuffle=True):
  '''read TFRecord file to get batch tensors for tensorflow models

  Returns:
    a tuple of batched tensors
  '''
  with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset([filename])
    # Parse the record into tensors
    dataset = dataset.map(parse_func)
    dataset = dataset.repeat(epoch)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    
    if shuffle:
      iterator = dataset.make_one_shot_iterator()
    else:
      iterator = dataset.make_initializable_iterator()
    return iterator

def shuf_and_write(filename):
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
