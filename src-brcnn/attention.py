'''
Q: Lq, d
K: L,  d   => Lq, dv
V: L,  dv


Attention(Q,K,V) = softmax(QK^T/sqrt(d))V

Lq,d x d,L => Lq,L x L,dv => Lq,dv
'''

import tensorflow as tf

def slice_batch_n(inputs, begin_n, size_n, dtype=tf.float32):
  '''
  Args
    inputs: [batch, length, dim]
    begin_n: a list of tensors of shape [batch]
    size_n: a list of tensors of shape [batch]
  '''
  size_total = tf.add_n(size_n)
  max_size = tf.reduce_max(size_total)
  batch_idx = tf.range(tf.shape(inputs)[0])

  # [batch, 1]
  begin_n = [tf.expand_dims(begin, axis=-1) for begin in begin_n]
  size_n = [tf.expand_dims(size, axis=-1) for size in size_n]

  # [batch, 2]
  begin_n = [tf.concat([begin, tf.zeros_like(begin)], axis=-1) for begin in begin_n]
  size_n = [tf.concat([size, -1*tf.ones_like(size)], axis=-1) for size in size_n]

  def map_fn(idx):
    slice_n = []
    for begin, size in zip(begin_n, size_n):
      slice = tf.slice(inputs[idx], begin[idx], size[idx])
      slice_n.append(slice)
    slice_n = tf.concat(slice_n, axis=0)
    pad = tf.pad(slice_n, [[0, max_size-size_total[idx]], [0, 0]])
    return pad

  return tf.map_fn(map_fn, batch_idx, dtype=dtype)

def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate=0.0,
                        reuse=None,
                        name=None):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    name: an optional string.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionaly returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):
    q, k, v = compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                          total_value_depth, reuse=reuse)

    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5

    x = dot_product_attention(q, k, v, bias, dropout_rate)

    x = combine_heads(x)
    x = tf.layers.dense(
        x, output_depth, use_bias=False, name="output_transform", reuse=reuse)
    return x


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                reuse=None):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: and integer
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  
  q = tf.layers.dense(query_antecedent, total_key_depth, use_bias=False, name="q", reuse=reuse)
  k = tf.layers.dense(memory_antecedent, total_key_depth, use_bias=False, name="k", reuse=reuse)
  v = tf.layers.dense(memory_antecedent, total_value_depth, use_bias=False, name="v", reuse=reuse)
  return q, k, v

def dot_product_attention(q,
                          k,
                          v,
                          bias=None,
                          dropout_rate=0.0,
                          name=None):
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")

    # FIXME: if is_train:
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    return tf.matmul(weights, v)

def combine_heads(x):
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  x_shape = shape_list(x)
  a, b = x_shape[-2:]
  return tf.reshape(x, x_shape[:-2] + [a * b])

def split_heads(x, num_heads):
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])

def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret