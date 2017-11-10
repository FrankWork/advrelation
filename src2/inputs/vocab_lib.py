# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generates vocabulary and term frequency files for datasets."""

import csv
import os
import re
from collections import defaultdict


# Dependency imports

import tensorflow as tf
from inputs.tfrecords import raw_dataset
# from adversarial_text.data import data_utils
# from adversarial_text.data import document_generators

# TODO: reduce vocab size
MAX_VOCAB_SIZE = None # 9006 + 1
EOS_TOKEN = '</s>'


flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags controlling input are in document_generators.py

flags.DEFINE_string('vocab_file', 'vocab.txt', 'Path to save vocab.txt.')
flags.DEFINE_string('vocab_freq_file', 'vocab_freq.txt', 'Path to save vocab_freq.txt.')

flags.DEFINE_integer('vocab_count_threshold', 1, 'The minimum number of '
                     'a word or bigram should occur to keep '
                     'it in the vocabulary.')

# flags.DEFINE_integer('vocab_size', None,
#                      'The size of the vocaburary. This value '
#                      'should be exactly same as the number of the '
#                      'vocabulary used in dataset. Because the last '
#                      'indexed vocabulary of the dataset preprocessed by '
#                      'my preprocessed code, is always <eos> and here we '
#                      'specify the <eos> with the the index.')


def split_by_punct(segment):
  """Splits str segment by punctuation, filters our empties and spaces."""
  return [s for s in re.split(r'\W+', segment) if s and not s.isspace()]

def gen_vocab(train_file):
  tf.logging.set_verbosity(tf.logging.INFO)

  vocab_freqs = defaultdict(int)

  # Fill vocabulary frequencies map 
  for example in raw_dataset(train_file):
    for token in example.sentence:
      vocab_freqs[token] += 1

  # Filter out low-occurring terms
  vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items()
                     if vocab_freqs[term] > FLAGS.vocab_count_threshold)

  # Sort by frequency
  ordered_vocab_freqs = sorted(
      vocab_freqs.items(), key= lambda item: item[1], reverse=True)

  # Limit vocab size
  if MAX_VOCAB_SIZE:
    ordered_vocab_freqs = ordered_vocab_freqs[:MAX_VOCAB_SIZE]

  # Add EOS token
  ordered_vocab_freqs.append((EOS_TOKEN, 1))

  # Write
  # tf.gfile.MakeDirs(FLAGS.output_dir)
  vocab_path = os.path.join(FLAGS.data_dir, FLAGS.vocab_file)
  vocab_freq_path = os.path.join(FLAGS.data_dir, FLAGS.vocab_freq_file)
  
  with open(vocab_path, 'w') as vocab_f:
    with open(vocab_freq_path, 'w') as freq_f:
      for word, freq in ordered_vocab_freqs:
        vocab_f.write('{}\n'.format(word))
        freq_f.write('{}\n'.format(freq))

def get_vocab_freqs():
  """Returns vocab frequencies.

  Returns:
    List of integers
  """
  path = os.path.join(FLAGS.data_dir, FLAGS.vocab_freq_file)

  with open(vocab_path) as freq_f:
      freqs =  [int(line.strip()) for line in freq_f]
  
  FLAGS.vocab_size = len(freqs)
  return freqs

def get_vocab():
  '''Return list of string
  '''
  vocab_path = os.path.join(FLAGS.data_dir, FLAGS.vocab_file)

  with open(vocab_path) as vocab_f:
      vocab =  [line.strip() for line in vocab_f]
  return vocab

def get_vocab_ids():
  '''return dict<string, int>
  '''
  vocab_path = os.path.join(FLAGS.data_dir, FLAGS.vocab_file)

  with open(vocab_path) as vocab_f:
      vocab_ids =  dict([(line.strip(), i) for i, line in enumerate(vocab_f)])
  return vocab_ids