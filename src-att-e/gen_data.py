import tensorflow as tf
from inputs import dataset, semeval_v2#, nyt2010

tf.logging.set_verbosity(tf.logging.INFO)

semeval_text = semeval_v2.SemEvalCleanedTextData()
# nyt_text = nyt2010.NYT2010CleanedTextData()

# length statistics
semeval_text.length_statistics()
# nyt_text.length_statistics()

# gen vocab
vocab_mgr = dataset.VocabMgr()
vocab_mgr.generate_vocab(semeval_text.tokens())

# trim embedding
# vocab_mgr.trim_pretrain_embedding()

# build SemEval record data
semeval_text.set_vocab_mgr(vocab_mgr)
semeval_record = semeval_v2.SemEvalCleanedRecordData(semeval_text)
semeval_record.generate_data()

# build nyt record data
# nyt_text.set_vocab_mgr(vocab_mgr)
# nyt_record = nyt2010.NYT2010CleanedRecordData(nyt_text)
# nyt_record.generate_data()

# INFO:tensorflow:(percent, quantile) [(50, 17.0), (70, 21.0), (80, 24.0), (90, 29.0), (95, 33.0), (98, 40.0), (100, 98.0)]
# INFO:tensorflow:(percent, quantile) [(50, 39.0), (70, 47.0), (80, 53.0), (90, 62.0), (95, 71.0), (98, 84.0), (100, 9621.0)]
# INFO:tensorflow:generate TFRecord data
# INFO:tensorflow:generate TFRecord data
# INFO:tensorflow:ignore 1361 examples
