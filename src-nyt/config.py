import tensorflow as tf


def _baisc_hparams():
  hparams =  tf.contrib.training.HParams(
    num_epochs          = 50,
    batch_size          = 100,
    word_embed_size     = 300, 
    pos_num             = 123,
    pos_dim             = 5,
    tune_word_embed     = False,
    tune_conv           = True,
    kernel_size         = 3,
    num_filters         = 310,
    num_classes         = None,
    l2_scale            = 0.001,
    dropout_rate        = 0.5,
    learning_rate       = 0.001,
    max_norm            = None,
    max_len             = 97,
    num_train_examples  = 0,
    num_test_examples   = 0,
    log_freq           = 1000,
    logdir              = "saved_models/",
    save_dir            = "nyt_model"
    )
  
  return hparams

def semeval_hparams():
  hparams = _baisc_hparams()
  hparams.num_classes = 19
  hparams.num_train_examples = 8000
  hparams.num_test_examples = 2717
  hparams.log_freq = 80
  hparams.learning_rate = 0.0001
  # hparams.tune_conv = False
  return hparams

def nyt_hparams():
  hparams = _baisc_hparams()
  hparams.num_classes = 53
  hparams.num_train_examples = 565208
  hparams.num_test_examples = 170497
  hparams.num_epochs = 5
  return hparams


class Config(object):
  out_dir = "data/generated"

  semeval_dir = "data/SemEval"
  # semeval_relations_file = 'relations.txt'
  semeval_train_file = "train.cln"
  semeval_test_file = "test.cln"
  semeval_train_record = "train.semeval.tfrecord"
  semeval_test_record = "test.semeval.tfrecord"
  # semeval_results_file = "results.txt"

  nyt_dir = "data/nyt2010"
  nyt_train_file = "train.cln"
  nyt_train_record = "train.nyt.tfrecord"
  nyt_test_file = "test.cln"
  nyt_test_record = "test.nyt.tfrecord"

  pretrain_embed_dir = 'data/pretrain'
  google_embed300_file = "embed300.google.npy"
  google_words_file = "google_words.lst"
  trimmed_embed300_file = "embed300.trim.npy"

  vocab_size = None
  vocab_file = "vocab.txt"


def get_config():
  config = Config()
  return config
