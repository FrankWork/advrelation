import os
import re
import numpy as np
import tensorflow as tf
from collections import defaultdict
from collections import namedtuple

flags = tf.app.flags

flags.DEFINE_string("vocab_file", "data/vocab.txt", 
                              "vocab of train and test data")

flags.DEFINE_string("google_embed300_file", 
                             "data/embed300.google.npy", 
                             "google news word embeddding")
flags.DEFINE_string("google_words_file", 
                             "data/google_words.lst", 
                             "google words list")
flags.DEFINE_string("trimmed_embed300_file", 
                             "data/embed300.trim.npy", 
                             "trimmed google embedding")

flags.DEFINE_string("senna_embed50_file", 
                             "data/embed50.senna.npy", 
                             "senna words embeddding")
flags.DEFINE_string("senna_words_file", 
                             "data/senna_words.lst", 
                             "senna words list")
flags.DEFINE_string("trimmed_embed50_file", 
                             "data/embed50.trim.npy", 
                             "trimmed senna embedding")
FLAGS = tf.app.flags.FLAGS # load FLAGS.word_dim

PAD_WORD = "<pad>"

# nltk.tokenize.regexp.WordPunctTokenizer
pattern = r'\d+|\w+|[^\w\s]+'
regexp = re.compile(pattern)

def wordpunct_tokenizer(line):
  line = re.sub(r"[^A-Za-z0-9]", " ", line)
  line = re.sub(r"\s{2,}", " ", line)

  return regexp.findall(line)

def write_vocab(vocab):
  '''collect words in sentence'''
  with open(FLAGS.vocab_file, 'w') as f:
    for w in sorted(list(vocab)):
      f.write('%s\n' % w)
    f.write('%s\n' % PAD_WORD)

def _load_vocab(vocab_file):
  # load vocab from file
  vocab = []
  with open(vocab_file) as f:
    for line in f:
      w = line.strip()
      vocab.append(w)

  return vocab

def _load_embedding(embed_file, words_file):
  embed = np.load(embed_file)

  words2id = {}
  words = _load_vocab(words_file)
  for id, w in enumerate(words):
    words2id[w] = id
  
  return embed, words2id

def maybe_trim_embeddings(vocab_file, 
                        pretrain_embed_file,
                        pretrain_words_file,
                        trimed_embed_file):
  '''trim unnecessary words from original pre-trained word embedding

  Args:
    vocab_file: a file of tokens in train and test data
    pretrain_embed_file: file name of the original pre-trained embedding
    pretrain_words_file: file name of the words list w.r.t the embed
    trimed_embed_file: file name of the trimmed embedding
  '''
  if not os.path.exists(trimed_embed_file):
    pretrain_embed, pretrain_words2id = _load_embedding(
                                              pretrain_embed_file,
                                              pretrain_words_file)
    word_embed=[]
    vocab = _load_vocab(vocab_file)
    for w in vocab:
      if w in pretrain_words2id:
        id = pretrain_words2id[w]
        word_embed.append(pretrain_embed[id])
      else:
        vec = np.random.normal(0,0.1,[FLAGS.word_dim])
        word_embed.append(vec)
    pad_id = -1
    word_embed[pad_id] = np.zeros([FLAGS.word_dim])

    word_embed = np.asarray(word_embed)
    np.save(trimed_embed_file, word_embed.astype(np.float32))
    
  
  word_embed, vocab2id = _load_embedding(trimed_embed_file, vocab_file)
  return word_embed, vocab2id

def map_words_to_id(raw_data, word2id):
  '''inplace convert sentence from a list of words to a list of ids
  Args:
    raw_data: a list of Raw_Example
    word2id: dict, {word: id, ...}
  '''
  pad_id = word2id[PAD_WORD]
  for raw_example in raw_data:
    for idx, word in enumerate(raw_example.sentence):
      raw_example.sentence[idx] = word2id[word]

    # pad the sentence to FLAGS.max_len
    pad_n = FLAGS.max_len - len(raw_example.sentence)
    raw_example.sentence.extend(pad_n*[pad_id])

def maybe_write_tfrecord(raw_data, filename):
  '''if the destination file is not exist on disk, convert the raw_data to 
  tf.trian.SequenceExample and write to file.

  Args:
    raw_data: a list of 'Raw_Example'
  '''
  if not os.path.exists(filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for raw_example in raw_data:
      example = build_sequence_example(raw_example)
      writer.write(example.SerializeToString())
    writer.close()

def read_tfrecord_to_batch(filename, epoch, batch_size, pad_value, shuffle=True):
  '''read TFRecord file to get batch tensors for tensorflow models

  Returns:
    a tuple of batched tensors
  '''
  with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset([filename])
    # Parse the record into tensors
    dataset = dataset.map(_parse_tfexample) 
    dataset = dataset.repeat(epoch)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=100)
    
    # [] for no padding, [None] for padding to maximum length
    # n = FLAGS.max_len
    # if FLAGS.model == 'mtl':
    #   # lexical, rid, direction, sentence, position1, position2
    #   padded_shapes = ([None,], [], [], [n], [n], [n])
    # else:
    #   # lexical, rid, sentence, position1, position2
    #   padded_shapes = ([None,], [], [n], [n], [n])
    # pad_value = tf.convert_to_tensor(pad_value)
    # dataset = dataset.padded_batch(batch_size, padded_shapes,
    #                                padding_values=pad_value)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch

def inputs():
  raw_train_data = load_raw_data(FLAGS.train_file)
  raw_test_data = load_raw_data(FLAGS.test_file)

  maybe_build_vocab(raw_train_data, raw_test_data, FLAGS.vocab_file)

  if FLAGS.word_dim == 50:
    word_embed, vocab2id = maybe_trim_embeddings(
                                        FLAGS.vocab_file,
                                        FLAGS.senna_embed50_file,
                                        FLAGS.senna_words_file,
                                        FLAGS.trimmed_embed50_file)
  elif FLAGS.word_dim == 300:
    word_embed, vocab2id = maybe_trim_embeddings(
                                        FLAGS.vocab_file,
                                        FLAGS.google_embed300_file,
                                        FLAGS.google_words_file,
                                        FLAGS.trimmed_embed300_file)

  # map words to ids
  map_words_to_id(raw_train_data, vocab2id)
  map_words_to_id(raw_test_data, vocab2id)

  # convert raw data to TFRecord format data, and write to file
  train_record = FLAGS.train_record
  test_record = FLAGS.test_record
  
  maybe_write_tfrecord(raw_train_data, train_record)
  maybe_write_tfrecord(raw_test_data, test_record)

  pad_value = vocab2id[PAD_WORD]
  train_data = read_tfrecord_to_batch(train_record, 
                              FLAGS.num_epochs, FLAGS.batch_size, 
                              pad_value, shuffle=True)
  test_data = read_tfrecord_to_batch(test_record, 
                              FLAGS.num_epochs, 2717, 
                              pad_value, shuffle=False)

  return train_data, test_data, word_embed


