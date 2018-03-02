import tensorflow as tf
import config as config_lib
from inputs import dataset, rc_dataset

tf.logging.set_verbosity(tf.logging.INFO)

config = config_lib.get_config()

semeval_text = rc_dataset.RCTextData(
      config.semeval_dir, config.semeval_train_file, config.semeval_test_file)
# semeval_text.length_statistics()

nyt_text = rc_dataset.NYTTextData(config.nyt_dir, config.nyt_train_file)
# nyt_text.length_statistics()

# gen vocab
vocab = dataset.Vocab(config.out_dir, config.vocab_file)
vocab.generate_vocab(semeval_text.tokens())

nyt_vocab = dataset.Vocab()
nyt_vocab.generate_vocab(nyt_text.tokens(), min_vocab_freq=2)

vocab.union(nyt_vocab)

# # trim embedding
embed = dataset.Embed(config.out_dir, config.trimmed_embed300_file, config.vocab_file)
google_embed = dataset.Embed(config.pretrain_embed_dir, 
                        config.google_embed300_file, config.google_words_file)
embed.trim_pretrain_embedding(google_embed)

# build SemEval record data
semeval_text.set_vocab(vocab)
nyt_text.set_vocab(vocab)

tf.logging.info('generate TFRecord data')
semeval_data = rc_dataset.RCRecordData(config.out_dir, 
      config.semeval_train_record, config.semeval_test_record)
semeval_data.generate_train_records([semeval_text.train_examples()])
semeval_data.generate_test_records([semeval_text.test_examples()])

nyt_data = rc_dataset.RCRecordData(config.out_dir, config.nyt_train_record)
nyt_data.generate_train_records([nyt_text.train_examples()])


# INFO:tensorflow:(percent, quantile) 
#  [(50, 18.0), (70, 22.0), (80, 25.0), (90, 29.0), (95, 34.0), (98, 40.0), (99, 46.0), (100, 97.0)]
# INFO:tensorflow:(percent, quantile) 
#  [(50, 39.0), (70, 47.0), (80, 53.0), (90, 62.0), (95, 71.0), (98, 84.0), (99, 95.0), (100, 9621.0)]

