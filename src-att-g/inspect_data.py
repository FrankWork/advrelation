import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

filename = "data/generated/test.semeval.tfrecord"

def parse_example(example):
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

  x = ent_pos[2] - ent_pos[1]

  return x, ent_pos

def padded_shapes():
  return ([]) 

dataset = tf.data.TFRecordDataset([filename])
# Parse the record into tensors
dataset = dataset.map(parse_example)
dataset = dataset.repeat(1)

# dataset = dataset.padded_batch(1, padded_shapes())

for idx, (x, ent_pos) in enumerate(tfe.Iterator(dataset)):
  flag = tf.less(x, 0.)
  
  sum = tf.reduce_sum(tf.cast(flag, tf.int32))
  
  if not tf.equal(sum, 0):
    # print(0)
    # print(x)
    print(idx, ent_pos)
    # exit()
  
  # exit()