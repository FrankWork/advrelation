import re
import os
import random

from inputs import dataset
import tensorflow as tf

class SemEvalCleanedTextData(dataset.TextDataset):
 
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
        sent = self.vocab_mgr.map_token_to_id(sent)
        length = len(sent)

        label = int(words[0])

        e1_first, e1_last  = (int(words[1]), int(words[2]))
        e2_first, e2_last  = (int(words[3]), int(words[4]))
        ent_pos = [e1_first, e1_last, e2_first, e2_last]

        pos1 = dataset.position_feature(e1_first, e1_last, length)
        pos2 = dataset.position_feature(e2_first, e2_last, length)

        yield {
          'label': [label], 'length': [length], 'ent_pos': ent_pos, 
          'sentence': sent, 'pos1': pos1, 'pos2': pos2}
    
class SemEvalCleanedRecordData(dataset.RecordDataset):

  def __init__(self, text_dataset, train_record_file=TRAIN_RECORD, 
               test_record_file=TEST_RECORD):
    super().__init__(text_dataset, 
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

    # transformed tensor
    # begin = tf.convert_to_tensor([ent_pos[0], ent_pos[2]])
    # size = tf.convert_to_tensor([ent_pos[1]-ent_pos[0]+1, ent_pos[3]-ent_pos[2]+1])

    return (label, length, ent_pos, 
            sentence, pos1, pos2)
  
  def padded_shapes(self):
    return ([], [], [4],
            [None], [None], [None])


def write_results(predictions):
  id2relation = []
  with open(os.path.join(DATA_DIR, LABEL_FILE)) as f:
    for id, line in enumerate(f):
      rel = line.strip()
      id2relation.append(rel)
  
  start_no = 8001
  with open(os.path.join(OUT_DIR, RESULTS_FILE), 'w') as f:
    for idx, id in enumerate(predictions):
      if idx < 2717:
        rel = id2relation[id]
        f.write('%d\t%s\n' % (start_no+idx, rel))
        