#coding:utf-8

data_dir = 'data'
ckpt_dir = 'checkpoint'
word_embd_dir = 'checkpoint/word_embd_wiki'
pos_embd_dir = 'checkpoint/pos_embd'
dep_embd_dir = 'checkpoint/dep_embd'
model_dir = 'checkpoint/modelv1'

entity2idfilepath='data/entity2id.txt'
entity2descfilepath='data/entity2desc_clean_final.txt'
train_triplet_filepath='data/train_triplet.txt'
test_triplet_filepath='data/test_triplet.txt'

num_epochs = 300
word_embd_dim = 300
pos_embd_dim = 25
dep_embd_dim = 75
relation_classes = 19
word_state_size = 200
convolution_state_size = 200
dep_state_size = 50
BATCH_SIZE = 100
lambda_l2 = 0.00002 #3

desc_lambda_l2=0.0012
starter_learning_rate = 0.003
dropout_keep_prob = 0.5
max_grad_norm=5.0
dropout_desc_keep=0.8
decay_steps = 480
decay_rate = 0.96
alpha = 0.66
win_size = 9
shuffle = True
pos = True
senna = False
directed = True


desc_filter_size=3 # the window size that used in entity description
max_entity_desc_length=75
desc_num_filters=80

belda=0.3
