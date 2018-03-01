import re
import os
import random

from inputs import dataset
from inputs import utils
import tensorflow as tf


class RCTextData(dataset.TextDataset):
 
  def __init__(self, data_dir, train_file, test_file):
    super().__init__(data_dir, train_file=train_file, test_file=test_file)

  def token_generator(self, file):
    with open(file) as f:
      for line in f:
        words = line.strip().split(' ')
        
        for token in words[5:]:
          yield token
  
  def get_length(self):
    length = []
    for file in [self.train_file, self.test_file]:
      if file is not None:
        with open(file) as f:
          for line in f:
            words = line.strip().split(' ')
            n = len(words[5:])
            length.append(n)
    return length

  def example_generator(self, file):
    with open(file) as f:
      for line in f:
        words = line.strip().split(' ')
        
        sent = words[5:]
        tf.logging.debug(sent)
        sent = self.vocab.encode(sent)
        length = len(sent)

        label_id = int(words[0])

        e1_first, e1_last  = (int(words[1]), int(words[2]))
        e2_first, e2_last  = (int(words[3]), int(words[4]))
        ent_pos = [e1_first, e1_last, e2_first, e2_last]

        pos1 = utils.position_feature(e1_first, e1_last, length)
        pos2 = utils.position_feature(e2_first, e2_last, length)

        yield {
          'label': [label_id], 'length': [length], 'ent_pos': ent_pos, 
          'sentence': sent, 'pos1': pos1, 'pos2': pos2}
    
class RCRecordData(dataset.RecordDataset):

  def __init__(self, out_dir, train_record_file, test_record_file):
    super().__init__(out_dir=out_dir,
      train_record_file=train_record_file, test_record_file=test_record_file)

  def parse_example(self, example):
    features = {
      "label": tf.FixedLenFeature([], tf.int64),
      "length": tf.FixedLenFeature([], tf.int64),
      "ent_pos": tf.FixedLenFeature([4], tf.int64),
      "sentence": tf.VarLenFeature(tf.int64),
      "pos1": tf.VarLenFeature(tf.int64),
      "pos2": tf.VarLenFeature(tf.int64),
    }
    feat_dict = tf.parse_single_example(example, features)

    # load from disk
    label = feat_dict['label']
    length = tf.cast(feat_dict['length'], tf.int32)
    ent_pos = tf.cast(feat_dict['ent_pos'], tf.int32)
    sentence = tf.sparse_tensor_to_dense(feat_dict['sentence'])
    pos1 = tf.sparse_tensor_to_dense(feat_dict['pos1'])
    pos2 = tf.sparse_tensor_to_dense(feat_dict['pos2'])

    return (label, length, ent_pos, 
            sentence, pos1, pos2)
  
  def padded_shapes(self):
    return ([], [], [4],
            [None], [None], [None])

class NYTTextData(RCTextData):
  def __init__(self, data_dir, train_file):
    super().__init__(data_dir, train_file=train_file, test_file=None)
  
  def example_generator(self, file):
    with open(file) as f:
      for line in f:
        words = line.strip().split(' ')
        
        sent = words[5:]
        if len(sent) > 97:
          continue
          
        tf.logging.debug(sent)
        sent = self.vocab.encode(sent)
        length = len(sent)

        label_id = int(words[0]) + 19

        e1_first, e1_last  = (int(words[1]), int(words[2]))
        e2_first, e2_last  = (int(words[3]), int(words[4]))
        ent_pos = [e1_first, e1_last, e2_first, e2_last]

        pos1 = utils.position_feature(e1_first, e1_last, length)
        pos2 = utils.position_feature(e2_first, e2_last, length)

        yield {
          'label': [label_id], 'length': [length], 'ent_pos': ent_pos, 
          'sentence': sent, 'pos1': pos1, 'pos2': pos2}