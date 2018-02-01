import re
import os
import random

from inputs import dataset
import tensorflow as tf

class TagConverter(object):
  def __init__(self, data_dir, class_label_file, tag_file):
    self.class_label = dataset.Label(data_dir, class_label_file)
    self.tags = dataset.Label(data_dir, tag_file)
  
  def extract_rel_from_label_text(self, label_id):
    e1_is_head = True
    rel_str = 'O'

    label_text = self.class_label.vocab[label_id]
    seg_idx = label_text.find('(')
    if seg_idx == -1:
      return rel_str, e1_is_head

    relation = label_text[:seg_idx]
    direction = label_text[seg_idx:]

    seg_idx = relation.find('-')
    rel_str = relation[0] + relation[seg_idx+1]
    if direction == '(e2,e1)':
      e1_is_head = False
    return rel_str, e1_is_head

  def build_tag(self, e_begin, e_end, rel_str, is_head=True):
    n = e_end - e_begin
    rel_role = '1' if is_head else '2'
    tag_template = '%s-%s-%s'
    tags_str = []

    if n == 1:
      tags_str.append(tag_template % ('S', rel_str, rel_role))
    elif n == 2:
      tags_str.append(tag_template % ('B', rel_str, rel_role))
      tags_str.append(tag_template % ('E', rel_str, rel_role))
    else:
      tags_str.append(tag_template % ('B', rel_str, rel_role))
      for _ in range(n-2):
        tags_str.append(tag_template % ('I', rel_str, rel_role))
      tags_str.append(tag_template % ('E', rel_str, rel_role))
    return tags_str

  def convert_to_tag(self, label_id, ent_indices, length):
    default_tag = 'O'
    tags_str = [default_tag for _ in range(length)]

    e1_begin, e1_end = ent_indices[0], ent_indices[1]+1
    e2_begin, e2_end = ent_indices[2], ent_indices[3]+1

    rel_str, e1_is_head = self.extract_rel_from_label_text(label_id)

    tags_ent = self.build_tag(e1_begin, e1_end, rel_str, e1_is_head)
    tags_str[e1_begin: e1_end] = tags_ent

    tags_ent = self.build_tag(e2_begin, e2_end, rel_str, not e1_is_head)
    tags_str[e2_begin: e2_end] = tags_ent
    tf.logging.debug(tags_str)

    return self.tags.encode(tags_str)

  


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

  def set_tag_converter(self, tag_converter):
    self.tag_converter = tag_converter

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
        ent_indices = [e1_first, e1_last, e2_first, e2_last]

        labels = self.tag_converter.convert_to_tag(label_id, ent_indices, length)
        tf.logging.debug(' ')

        assert len(sent) == len(labels)

        yield {
          'labels': labels, 'length': [length], 'sentence': sent}
    
class SemEvalCleanedRecordData(dataset.RecordDataset):

  def __init__(self, text_dataset, out_dir, train_record_file, test_record_file):
    super().__init__(text_dataset, out_dir=out_dir,
      train_record_file=train_record_file, test_record_file=test_record_file)

  def parse_example(self, example):
    features = {
        "length": tf.FixedLenFeature([], tf.int64),
        "sentence": tf.VarLenFeature(tf.int64),
        "labels": tf.VarLenFeature(tf.int64)
    }
    feat_dict = tf.parse_single_example(example, features)

    # load from disk
    length = tf.cast(feat_dict['length'], tf.int32)
    sentence = tf.sparse_tensor_to_dense(feat_dict['sentence'])
    labels = tf.sparse_tensor_to_dense(feat_dict['labels'])

    return length, sentence, labels
  
  def padded_shapes(self):
    return ([], [None], [None])


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
        