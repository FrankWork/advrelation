import random
import tensorflow as tf

def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    # record is of <class 'bytes'>
    records.append(record)
  return records

def write_records(records, out_filename):
  writer = tf.python_io.TFRecordWriter(out_filename)
  for count, record in enumerate(records):
    writer.write(record)
  writer.close()

def shuf_and_write(filename):
  records = read_records(filename)
  random.shuffle(records)
  write_records(records, filename)


shuf_and_write("data/generated/train.imdb.tfrecord")
shuf_and_write("data/generated/test.imdb.tfrecord")

shuf_and_write("data/generated/train.semeval.tfrecord")
shuf_and_write("data/generated/test.semeval.tfrecord")