import tensorflow as tf

from inputs import dataset

DATA_DIR = 'data/nyt2010'
UNSUP_FILE = 'unsupervised.txt'
UNSUP_RECORD_FILE = 'unsup.nyt.tfrecord'
MAX_LEN = 98

class NYT2010CleanedTextData(dataset.TextDataset):

  def __init__(self, data_dir=DATA_DIR, max_len=MAX_LEN, unsup_file=UNSUP_FILE):
    super().__init__(data_dir, max_len=max_len, unsup_file=unsup_file)

  def token_generator(self, file):
    with open(file) as f:
      for line in f:
        words = line.strip().split(' ')
        
        for token in words[5:]:
          yield token

  def get_length(self):
    length = []
    with open(self.unsup_file) as f:
      for line in f:
        words = line.strip().split(' ')
        n = len(words[5:])
        length.append(n)
    return length

  def example_generator(self, file):
    n=0
    with open(file) as f:
      for line in f:
        words = line.strip().split(' ')
        
        sent = words[5:]
        sent = self.vocab_mgr.map_token_to_id(sent)
        length = len(sent)
        if length > MAX_LEN or length==0:
          n+=1
          continue


        label = int(words[0])

        e1_first, e1_last  = (int(words[1]), int(words[2]))
        e2_first, e2_last  = (int(words[3]), int(words[4]))

        pos1 = dataset.position_feature(e1_first, e1_last, length)
        pos2 = dataset.position_feature(e2_first, e2_last, length)

        yield {
          'label': [label], 'length': [length], 'sentence': sent, 
          'pos1': pos1, 'pos2': pos2}
      tf.logging.info('ignore %d examples' % n)

 
class NYT2010CleanedRecordData(dataset.RecordDataset):

  def __init__(self, text_dataset, unsup_record_file=UNSUP_RECORD_FILE):
    super().__init__(text_dataset, unsup_record_file=unsup_record_file)

  def parse_example(self, example):
    features = {
        "label": tf.FixedLenFeature([], tf.int64),
        "length": tf.FixedLenFeature([], tf.int64),
        "sentence": tf.VarLenFeature(tf.int64),
        "pos1": tf.VarLenFeature(tf.int64),
        "pos2": tf.VarLenFeature(tf.int64),
    }
    feat_dict = tf.parse_single_example(example, features)
    label = feat_dict['label']
    length = feat_dict['length']
    sentence = tf.sparse_tensor_to_dense(feat_dict['sentence'])
    pos1 = tf.sparse_tensor_to_dense(feat_dict['pos1'])
    pos2 = tf.sparse_tensor_to_dense(feat_dict['pos2'])
    return label, length, sentence, pos1, pos2

  
  def padded_shapes(self):
    return ([], [], [None], [None], [None])
