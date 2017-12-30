from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers

import tensorflow as tf

def residual_block(x, hparams):
  """A stack of convolution blocks with residual connection."""
  k = (hparams.kernel_height, hparams.kernel_width)
  dilations_and_kernels = [((1, 1), k) for _ in xrange(3)]
  y = common_layers.subseparable_conv_block(
      x,
      hparams.hidden_size,
      dilations_and_kernels,
      padding="SAME",
      separability=0,
      name="residual_block")
  x = common_layers.layer_norm(x + y, hparams.hidden_size, name="lnorm")
  return tf.nn.dropout(x, 1.0 - hparams.dropout)


def xception_internal(inputs, hparams):
  """Xception body."""
  with tf.variable_scope("xception"):
    cur = inputs
    for i in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % i):
        cur = residual_block(cur, hparams)
    return cur

@registry.register_model
class RelationAdv(t2t_model.T2TModel):

  # def bottom(self, features):
  #   print('in bottom:')
  #   for k, v in features.items():
  #     print(k, v.shape)
  #   # exit()
  #   return features

  def body(self, features):
    '''
    Args:
      features: dict<string, tensor>, `inputs` tensor is the results of 
                embedding lookup of the origin `inputs` tensor. Lookup operation
                is done in `self.bottom`
    '''
    print('in body')
    for k, v in features.items():
      print(k, v.shape)
    exit()
    # return 'hello world'

@registry.register_hparams
def relation_adv_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.max_input_seq_length = 0
  hparams.batch_size = 100
  hparams.hidden_size = 50 # word embedding dim
  hparams.dropout = 0.2
  hparams.symbol_dropout = 0.2
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 8
  hparams.kernel_height = 3
  hparams.kernel_width = 3
  hparams.learning_rate_decay_scheme = "exp50k"
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 3.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  return hparams