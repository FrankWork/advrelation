
def get_hparams():
  import tensorflow as tf

  hparams =  tf.contrib.training.HParams(
    word_embed_size        = 300, 
    hidden_size            = 300,
    num_tags               = 0,
    l2_scale               = 0.001,
    dropout_rate           = 0.5,
    learning_rate          = 0.001,
    )
  
  return hparams


