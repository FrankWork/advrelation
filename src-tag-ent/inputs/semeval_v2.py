import re
import os
import random

from inputs import dataset
import tensorflow as tf


class SemEvalCleanedTextData(dataset.TextDataset):
 
  def __init__(self, data_dir, train_file, test_file):
    self.tag_converter = None
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

  def set_tags_encoder(self, tags_encoder):
    self.tags_encoder = tags_encoder

  def _build_tags(self, ent_indices, length, direction):
    e1_begin, e1_end, e2_begin, e2_end = ent_indices
    default_tags = 'O'
    tags = [default_tags for _ in range(length)]

    n = e1_end - e1_begin
    e1_tags = []
    e1_role = 'h' if direction==0 else 't'
    if n == 1:
      e1_tags.append('S-'+e1_role)
    elif n == 2:
      e1_tags.append('B-'+e1_role)
      e1_tags.append('E-'+e1_role)
    else:
      e1_tags.append('B-'+e1_role)
      for _ in range(n-2):
        e1_tags.append('I-'+e1_role)
      e1_tags.append('E-'+e1_role)
    tags[e1_begin:e1_end] = e1_tags

    n = e2_end - e2_begin
    e2_tags = []
    e2_role = 't' if direction==0 else 'h'
    if n == 1:
      e2_tags.append('S-'+e2_role)
    elif n == 2:
      e2_tags.append('B-'+e2_role)
      e2_tags.append('E-'+e2_role)
    else:
      e2_tags.append('B-'+e2_role)
      for _ in range(n-2):
        e2_tags.append('I-'+e2_role)
      e2_tags.append('E-'+e2_role)
    tags[e2_begin:e2_end] = e2_tags


    return tags

  def example_generator(self, file):
    with open(file) as f:
      for line in f:
        words = line.strip().split(' ')
        
        sent_toks = words[5:]
        sent = self.vocab.encode(sent_toks)
        length = len(sent)

        label = int(words[0])
        direction = label % 2 # 0 (e1,e2); 1 (e2,e1)
        label = label // 2

        e1_begin, e1_end  = (int(words[1]), int(words[2])+1)
        e2_begin, e2_end  = (int(words[3]), int(words[4])+1)
        ent_indices = [e1_begin, e1_end, e2_begin, e2_end]

        tags = self._build_tags(ent_indices, length, direction)
        tf.logging.debug(list(zip(sent_toks, tags)))
        tags = self.tags_encoder.encode(tags)

        assert len(sent) == len(tags)

        yield {
          'label': [label], 'length': [length], 'sentence': sent, 'tags': tags}
    
class SemEvalCleanedRecordData(dataset.RecordDataset):

  def __init__(self, text_dataset, out_dir, train_record_file, test_record_file):
    super().__init__(text_dataset, out_dir=out_dir,
      train_record_file=train_record_file, test_record_file=test_record_file)

  def parse_example(self, example):
    features = {
        "label": tf.FixedLenFeature([], tf.int64),
        "length": tf.FixedLenFeature([], tf.int64),
        "sentence": tf.VarLenFeature(tf.int64),
        "tags": tf.VarLenFeature(tf.int64)
    }
    feat_dict = tf.parse_single_example(example, features)

    # load from disk
    label = feat_dict['label']
    length = feat_dict['length']
    sentence = tf.sparse_tensor_to_dense(feat_dict['sentence'])
    tags = tf.sparse_tensor_to_dense(feat_dict['tags'])

    return label, length, sentence, tags
  
  def padded_shapes(self):
    return ([], [], [None], [None])


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
        