from tensor2tensor.utils import modality
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils as eu

import tensorflow as tf


@registry.register_symbol_modality("embed")
class EmbedSymbolModality(modality.Modality):
  """Used for label data."""

  def __init__(self, model_hparams, vocab_size=None):
    self._model_hparams = model_hparams
    self._vocab_size = vocab_size

  @property
  def name(self):
    return "embed_symbol_modality_%d_%d" % (self.vocab_size, self.dense_size)
  
  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def dense_size(self):
    return self._body_input_depth

  @property
  def multiplier(self):
    if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
      return self.dense_size**0.5

    return 1.0

  def _embedding(self, x, reuse=None):
    with tf.variable_scope(self.name):
      return common_layers.embedding(x, self.vocab_size, self.dense_size,
                                    reuse=reuse, multiplier=self.multiplier)

  def bottom(self, x):
    with tf.variable_scope(self.name):
      try:
        return self._embedding(x, reuse=True)
      except ValueError:
        return self._embedding(x, reuse=None)
      
@registry.register_symbol_modality("position")
class PositionSymbolModality(EmbedSymbolModality):
  """Used for label data."""

  def __init__(self, model_hparams, vocab_size=None):
    self._model_hparams = model_hparams
    # self._vocab_size = vocab_size

  @property
  def name(self):
    return "position_symbol_modality_%d_%d" % (self.vocab_size, self.dense_size)

  @property
  def vocab_size(self):
    return 123

  @property
  def dense_size(self):
    return 5

  def bottom(self, x):
    with tf.variable_scope(self.name):
      return common_layers.embedding(
          x,
          self.vocab_size,
          self.dense_size,
          multiplier=1.0)