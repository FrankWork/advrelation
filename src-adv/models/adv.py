import tensorflow as tf


def scale_l2(x, norm_length=5):
  # shape(x) = (batch, num_timesteps, d)
  # Divide x by max(abs(x)) for a numerically stable L2 norm.
  # 2norm(x) = a * 2norm(x/a)
  # Scale over the full sequence, dims (1, 2)
  alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
  l2_norm = alpha * tf.sqrt(
      tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
  x_unit = x / l2_norm
  return norm_length * x_unit

def mask_by_length(t, length):
  """Mask t, 3-D [batch, time, dim], by length, 1-D [batch,]."""
  maxlen = t.get_shape().as_list()[1]

  # Subtract 1 from length to prevent the perturbation from going on 'eos'
  mask = tf.sequence_mask(length, maxlen=maxlen)
  mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
  # shape(mask) = (batch, num_timesteps, 1)
  return t * mask

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm

def normalize_embed(self, emb, vocab_freqs):
    vocab_freqs = tf.constant(
          vocab_freqs, dtype=tf.float32, shape=[self.vocab_size, 1])
    weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev

def adv_example(self, input, loss):
    grad, = tf.gradients(
        loss,
        input,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = scale_l2(grad)
    return input + perturb

def adversarial_loss(self, loss, lexical, sentence, pos1, pos2, labels):
    adv_lexical = self.adv_example(lexical, loss)
    adv_sentence = self.adv_example(sentence, loss)
    _, loss = self.xentropy_logits_and_loss(adv_lexical, adv_sentence, pos1, pos2, labels)
    return loss

def kl_divergence_with_logits(self, q_logit, p_logit):
    # https://github.com/takerum/vat_tf
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    kl = qlogq - qlogp
    return kl

def virtual_adversarial_loss(self, logits, lexical, length, sentence, pos1, pos2):
    # Stop gradient of logits. See https://arxiv.org/abs/1507.00677 for details.
    logits = tf.stop_gradient(logits)

    # Initialize perturbation with random noise.
    d_sent = tf.random_normal(shape=tf.shape(sentence))
    d_lex = tf.random_normal(shape=tf.shape(lexical))

    # Perform finite difference method and power iteration.
    # See Eq.(8) in the paper http://arxiv.org/pdf/1507.00677.pdf,
    # Adding small noise to input and taking gradient with respect to the noise
    # corresponds to 1 power iteration.
    agg_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
    for _ in range(NUM_POWER_ITER):
      d_sent = scale_l2(mask_by_length(d_sent, length), SMALL_CONSTANT) 
      d_lex = scale_l2(d_lex, SMALL_CONSTANT)
      vadv_sent = sentence + d_sent
      vadv_lex = lexical + d_lex
      d_logits, _ = self.xentropy_logits_and_loss(vadv_lex, vadv_sent, pos1, pos2, None)

      kl = self.kl_divergence_with_logits(logits, d_logits)
      d_sent, = tf.gradients(kl, d_sent, aggregation_method=agg_method)
      d_sent = tf.stop_gradient(d_sent)
      d_lex, = tf.gradients(kl, d_lex, aggregation_method=agg_method)
      d_lex = tf.stop_gradient(d_lex)

    vadv_sent = sentence + scale_l2(d_sent)
    vadv_lex = lexical + scale_l2(d_lex)
    vadv_logits, _ = self.xentropy_logits_and_loss(vadv_lex, vadv_sent, pos1, pos2, None)

    return self.kl_divergence_with_logits(logits, vadv_logits)