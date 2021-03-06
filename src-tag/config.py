
def _get_hparams():
  import tensorflow as tf

  hparams =  tf.contrib.training.HParams(
    num_epochs             = 50,
    batch_size             = 100,
    word_embed_size        = 300, 
    tune_word_embed        = False,
    hidden_size            = 300,
    num_tags               = 81,
    l2_scale               = 0.001,
    dropout_rate           = 0.5,
    learning_rate          = 0.001,
    max_norm               = None
    )
  
  return hparams

class Config(object):
  
  hparams = _get_hparams()
  
  logdir = "saved_models/"
  save_dir = "tag_model"
  
  out_dir = "data/generated"

  semeval_dir = "data/SemEval"
  semeval_relations_file = 'relations.txt'
  semeval_tags_file = 'tags.txt'
  semeval_train_file = "train.cln"
  semeval_test_file = "test.cln"
  semeval_train_record = "train.semeval.tfrecord"
  semeval_test_record = "test.semeval.tfrecord"
  semeval_results_file = "results.txt"

  pretrain_embed_dir = 'data/pretrain'
  google_embed300_file = "embed300.google.npy"
  google_words_file = "google_words.lst"
  trimmed_embed300_file = "embed300.trim.npy"

  vocab_size = None
  vocab_file = "vocab.txt"

_config = Config()

def get_config():
  return _config
