import tensorflow as tf
from decode import *

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()


batch = 4
max_len = 10
dim = 2
hidden_size = 3

inputs = tf.random_normal([max_len, batch, dim])
lengths = tf.convert_to_tensor([4,5,6,7])

c = tf.zeros([batch, hidden_size])
h = tf.zeros([batch, hidden_size])
state = TagLSTMStateTuple(c, h, tf.zeros_like(c))

output = decode(inputs, state, lengths, hidden_size)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  t = sess.run(output)
  print(t.shape)
  # print(t)