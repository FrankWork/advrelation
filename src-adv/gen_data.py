import tensorflow as tf
from inputs import dataset, semeval_v2#nyt2010, 

tf.logging.set_verbosity(tf.logging.INFO)

semeval_text = semeval_v2.SemEvalCleanedTextData()

# length statistics
semeval_text.length_statistics()


# gen vocab
vocab_mgr = dataset.VocabMgr()
vocab_mgr.generate_vocab(semeval_text.tokens())

# trim embedding
vocab_mgr.trim_pretrain_embedding()

# build SemEval record data
semeval_text.set_vocab_mgr(vocab_mgr)
semeval_record = semeval_v2.SemEvalCleanedRecordData(semeval_text)
semeval_record.generate_data()
