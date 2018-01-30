
def _get_hparams():
  import tensorflow as tf

  hparams =  tf.contrib.training.HParams(
    word_embed_size        = 300, 
    tune_word_embed        = False,
    hidden_size            = 300,
    num_tags               = 0,
    l2_scale               = 0.001,
    dropout_rate           = 0.5,
    learning_rate          = 0.001,
    )
  
  return hparams

class Config(object):
  
  hparams = _get_hparams()
  logdir = "saved_models/"
  save_dir = "tag_model"

_config = Config()

def get_config():
  return _config
