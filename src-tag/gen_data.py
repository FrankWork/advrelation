import tensorflow as tf
import config as config_lib
from inputs import dataset, semeval_v2

tf.logging.set_verbosity(tf.logging.INFO)

config = config_lib.get_config()

semeval_text = semeval_v2.SemEvalCleanedTextData(
      config.semeval_dir, config.semeval_train_file, config.semeval_test_file)

# length statistics
semeval_text.length_statistics()

# gen vocab
vocab = dataset.Vocab(config.out_dir, config.vocab_file)
vocab.generate_vocab(semeval_text.tokens())

# trim embedding
embed = dataset.Embed(config.out_dir, config.trimmed_embed300_file, config.vocab_file)
google_embed = dataset.Embed(config.pretrain_embed_dir, 
                        config.google_embed300_file, config.google_words_file)
embed.trim_pretrain_embedding(google_embed)

# build SemEval record data
semeval_text.set_vocab(vocab)
tag_converter = semeval_v2.TagConverter(config.semeval_dir, 
                      config.semeval_relations_file, config.semeval_tags_file)
semeval_text.set_tag_converter(tag_converter)
semeval_record = semeval_v2.SemEvalCleanedRecordData(semeval_text,
        config.out_dir, config.semeval_train_record, config.semeval_test_record)
semeval_record.generate_data()


# INFO:tensorflow:(percent, quantile) [(50, 18.0), (70, 22.0), (80, 25.0), 
#                              (90, 29.0), (95, 34.0), (98, 40.0), (100, 97.0)]
# INFO:tensorflow:generate vocab to data/generated/vocab.txt
# INFO:tensorflow:trim embedding to data/generated/embed300.trim.npy
# INFO:tensorflow:generate TFRecord data
