data_dir = 'data'
ckpt_dir = 'checkpoint'
word_embd_dir = 'checkpoint/word_embd_wiki'
pos_embd_dir = 'checkpoint/pos_embd'
dep_embd_dir = 'checkpoint/dep_embd'
model_dir = 'checkpoint/modelv1'

num_epochs = 50
word_embd_dim = 300
pos_embd_dim = 25
dep_embd_dim = 75
relation_classes = 19
word_state_size = 200
convolution_state_size = 200
dep_state_size = 50
BATCH_SIZE = 100
lambda_l2 = 0.00001
starter_learning_rate = 0.001
dropout_keep_prob = 0.5
decay_steps = 480
decay_rate = 0.96
alpha = 0.65
win_size = 9
shuffle = True
pos = True
senna = False
directed = True
