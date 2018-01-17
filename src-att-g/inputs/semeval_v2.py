import re
import os
import random

from inputs import dataset
import tensorflow as tf

DATA_DIR = "data/SemEval"
OUT_DIR = "data/generated"

TRAIN_FILE = "train.cln"
TEST_FILE = "test.cln"
LABEL_FILE = "relations.txt"

TRAIN_RECORD = "train.semeval.tfrecord"
TEST_RECORD = "test.semeval.tfrecord"
RESULTS_FILE = "results.txt"

class SemEvalCleanedTextData(dataset.TextDataset):
 
  def __init__(self, data_dir=DATA_DIR, train_file=TRAIN_FILE, test_file=TEST_FILE):
    super().__init__(data_dir, 
          train_file=train_file, test_file=test_file)

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
    length = feat_dict['length']
    ent_pos = tf.cast(feat_dict['ent_pos'], tf.int32)
    sentence = tf.sparse_tensor_to_dense(feat_dict['sentence'])
    pos1 = tf.sparse_tensor_to_dense(feat_dict['pos1'])
    pos2 = tf.sparse_tensor_to_dense(feat_dict['pos2'])

    # extract entities and context from tensor
    # --------e1.x++++e1.y-------e2.x****e2.y--------
    begin1 = [ent_pos[0]]
    size1 = [ent_pos[1]+1-ent_pos[0] ]
    ent1 = tf.slice(sentence, begin1, size1)
    ent1_pos1 = tf.slice(pos1, begin1, size1)
    ent1_pos2 = tf.slice(pos2, begin1, size1)

    begin2 = [ent_pos[2] ]
    size2 = [ent_pos[3]+1-ent_pos[2] ]
    ent2 = tf.slice(sentence, begin2, size2)
    ent2_pos1 = tf.slice(pos1, begin2, size2)
    ent2_pos2 = tf.slice(pos2, begin2, size2)

    
    begin1 = [0]
    size1 = [ent_pos[0] ]
    cont1 = tf.slice(sentence, begin1,  size1)

    begin2 = [ent_pos[1]+1]
    size2 = [ent_pos[2]-ent_pos[1]-1]
    cont2 = tf.slice(sentence, begin2, size2)

    # begin3 = [ent_pos[3]+1]
    # size3 = [length-ent_pos[3]-1]
    # cont3 = tf.slice(sentence, begin3, size3)

    # context = tf.concat([cont1, cont2, cont3], axis=0)
    context = cont2


    return (label, length, ent_pos, 
            sentence, pos1, pos2, 
            ent1, ent1_pos1, ent1_pos2,
            ent2, ent2_pos1, ent2_pos2,
            context)

  
  def padded_shapes(self):
    return ([], [], [4], 
            [None], [None], [None], 
            [None], [None], [None], 
            [None], [None], [None],
            [None])


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
        