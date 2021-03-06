import sys, os, pickle as pickle
import tensorflow as tf
import numpy as np
#from sklearn.metrics import f1_score
import subprocess
from config import *
import gensim
import datetime
import math
from desc_utils import *

embeddings = []
word2id = {}
id2word = {}
pad_word = "<pad>"
word2id[pad_word] = 0
id2word[0] = pad_word
embeddings.append([0 for i in range(word_embd_dim)])

if not senna:
    f = open(data_dir + '/extract_embed_300d.pkl', 'rb')
    word2vec = pickle.load(f)
    # word2vec = gensim.models.KeyedVectors.load_word2vec_format(data_dir + '/vectors-3.bin.gz', binary=True)

    wid = 1
    for word in word2vec:
        embeddings.append(word2vec[word])
        word2id[word] = wid
        id2word[wid] = word
        wid += 1
else:
    wordlist=open(data_dir + '/senna/words.lst',"r").readlines()
    allword_embedding=open(data_dir + '/senna/embeddings.txt',"r").readlines()

    for wid in range(len(wordlist)):
        word=wordlist[wid].strip()
        one_embedding=allword_embedding[wid].strip().split()
        embeddings.append(one_embedding)
        word2id[word]=wid+1

# word2id = dict((w, i+2) for i,w in enumerate(word2vec.vocab)) # word emb rand
# id2word = dict((i+2, w) for i,w in enumerate(word2vec.vocab))

pos_tags_vocab = []
for line in open(data_dir + '/pos_tags.txt'):
        pos_tags_vocab.append(line.strip())

dep_vocab = []
for line in open(data_dir + '/dependency_types.txt'):
    dep_vocab.append(line.strip())

relation_vocab = []
for line in open(data_dir + '/relation_types.txt'):
    relation_vocab.append(line.strip())

rel2id = dict((w, i) for i,w in enumerate(relation_vocab))
id2rel = dict((i, w) for i,w in enumerate(relation_vocab))

pos_tag2id = dict((w, i+1) for i,w in enumerate(pos_tags_vocab))
id2pos_tag = dict((i+1, w) for i,w in enumerate(pos_tags_vocab))

if not directed:
    dep2id = dict((w, i+1) for i,w in enumerate(dep_vocab))
    id2dep = dict((i+1, w) for i,w in enumerate(dep_vocab))
else:
    dep2id = dict(('l'+w, i+1) for i,w in enumerate(dep_vocab))
    id2dep = dict((i+1, 'l'+w) for i,w in enumerate(dep_vocab))
    dep2id.update(dict(('r'+w, len(dep_vocab)+i+1) for i,w in enumerate(dep_vocab)))
    id2dep.update(dict((len(dep_vocab)+i+1, 'r'+w) for i,w in enumerate(dep_vocab)))

pos_tag2id[pad_word] = 0
id2pos_tag[0] = pad_word

dep2id[pad_word] = 0
id2dep[0] = pad_word

pos_tag2id['OTH'] = len(pos_tag2id)
id2pos_tag[len(pos_tag2id)] = 'OTH'

dep2id['OTH'] = len(dep2id)
id2dep[len(dep2id)] = 'OTH'

JJ_pos_tags = ['JJ', 'JJR', 'JJS']
NN_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
RB_pos_tags = ['RB', 'RBR', 'RBS']
PRP_pos_tags = ['PRP', 'PRP$']
VB_pos_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
_pos_tags = ['CC', 'CD', 'DT', 'IN']

def pos_tag(x):
    if x in JJ_pos_tags:
        return pos_tag2id['JJ']
    if x in NN_pos_tags:
        return pos_tag2id['NN']
    if x in RB_pos_tags:
        return pos_tag2id['RB']
    if x in PRP_pos_tags:
        return pos_tag2id['PRP']
    if x in VB_pos_tags:
        return pos_tag2id['VB']
    if x in _pos_tags:
        return pos_tag2id[x]
    else:
        return 0

if not directed:
    f = open(data_dir + '/train_paths', 'rb')
    f_test = open(data_dir + '/test_paths', 'rb')
else:
    f = open(data_dir + '/vdir/directed_train_paths', 'rb')
    f_test = open(data_dir + '/vdir/directed_test_paths', 'rb')

word_p, dep_p, pos_p = pickle.load(f)
f.close()
word_p_test, dep_p_test, pos_p_test = pickle.load(f_test)
f_test.close()

init_vocab_count = len(word2id)
unknown_token = "UNKNOWN_TOKEN"
word2id[unknown_token] = init_vocab_count
id2word[len(word2id)] = unknown_token

relations = []
for line in open(data_dir + '/train_relations.txt'):
    relations.append(line.strip().split()[1])

length = len(word_p)
num_batches = int(math.ceil(length/BATCH_SIZE))

path_len = np.array([len(w) for w in word_p], dtype=int)
max_len_path = np.max(path_len)
path_len_test = np.array([len(w) for w in word_p_test], dtype=int)
max_len_path_test = np.max(path_len_test)
max_len_path = max(max_len_path, max_len_path_test)

word_p_ids = np.zeros([length, max_len_path],dtype=int)
pos_p_ids = np.zeros([length, max_len_path],dtype=int)
dep_p_ids = np.zeros([length, max_len_path],dtype=int)
dep_p_ids_reverse = np.zeros([length, max_len_path],dtype=int)
rel_ids = np.array([rel2id[rel] for rel in relations])

for i in range(length):
    for j, w in enumerate(word_p[i]):
        w = w.lower()
        if w not in word2id:
            word2id[w] = len(word2id)
            id2word[len(word2id)] = w
        word_p_ids[i][j] = word2id[w]
    for j, p in enumerate(pos_p[i]):
        if p not in pos_tag2id:
            pos_tag2id[p] = len(pos_tag2id)
            id2pos_tag[len(pos_tag2id)] = p
        pos_p_ids[i][j] = pos_tag(p)
    for j, d in enumerate(dep_p[i]):
        if d not in dep2id:
            dep2id[d] = len(dep2id)
            id2dep[len(dep2id)] = d
        dep_p_ids[i][j] = dep2id[d]
        if directed:
            if d.startswith('l'):
                dep_p_ids_reverse[i][j] = dep2id['r'+d[1:]]
            elif d.startswith('r'):
                dep_p_ids_reverse[i][j] = dep2id['l'+d[1:]]
            else:
                dep_p_ids_reverse[i][j] = dep2id[d]



############################################
#############add desc contents
#############################################

##load entity2id file
entity2id = load_entity2id(entity2idfilepath)
#load entity's description file
entityid2desc_dict=load_entity_desc(entity2descfilepath,entity2id)
pad_descs(entityid2desc_dict, max_entity_desc_length, pad_word)
#enid2desc_id = convert_desc_into_index(entityid2desc_dict, word2id,id2word)

train_en1_to_desc_id, train_en2_to_desc_id = get_entity_to_descid(train_triplet_filepath, entity2id, entityid2desc_dict, word2id,id2word,is_training=True)

embeddings = np.asarray(embeddings)
miss_embe_count = len(word2id)-init_vocab_count
miss_embeddings = np.random.uniform(-0.01,0.01,[miss_embe_count, word_embd_dim] )
embeddings = np.vstack((embeddings, miss_embeddings))

relations_test = []
for line in open(data_dir + '/test_relations.txt'):
    relations_test.append(line.strip().split()[0])

length_test = len(word_p_test)
num_batches_test = int(math.ceil(length_test/BATCH_SIZE))

for i in range(length_test):
    for j, word in enumerate(word_p_test[i]):
        word = word.lower()
        word_p_test[i][j] = word if word in word2id else unknown_token
    for l, d in enumerate(dep_p_test[i]):
        dep_p_test[i][l] = d if d in dep2id else 'OTH'
    for l, p in enumerate(pos_p_test[i]):
        pos_p_test[i][l] = p if p in pos_tag2id else 'OTH'

word_p_ids_test = np.zeros([length_test, max_len_path],dtype=int)
pos_p_ids_test = np.zeros([length_test, max_len_path],dtype=int)
dep_p_ids_test = np.zeros([length_test, max_len_path],dtype=int)
dep_p_ids_test_reverse = np.zeros([length_test, max_len_path],dtype=int)
rel_ids_test = np.array([rel2id[rel] for rel in relations_test])

for i in range(length_test):
    for j, w in enumerate(word_p_test[i]):
        word_p_ids_test[i][j] = word2id[w]
    for j, p in enumerate(pos_p_test[i]):
        pos_p_ids_test[i][j] = pos_tag(p)
    for j, d in enumerate(dep_p_test[i]):
        dep_p_ids_test[i][j] = dep2id[d]
        if directed:
            if d.startswith('l'):
                dep_p_ids_test_reverse[i][j] = dep2id['r'+d[1:]]
            elif d.startswith('r'):
                dep_p_ids_test_reverse[i][j] = dep2id['l'+d[1:]]
            else:
                dep_p_ids_test_reverse[i][j] = dep2id[d]


#####################
# add desc contents
test_en1_to_desc_id, test_en2_to_desc_id = get_entity_to_descid(test_triplet_filepath, entity2id, entityid2desc_dict, word2id,id2word,is_training=False)

batch_size = BATCH_SIZE
word_vocab_size = len(word2id)
pos_vocab_size = len(pos_tag2id)
dep_vocab_size = len(dep2id)


################
# attention related
################
train_ent_pos = []
with open('train_ent_pos.txt') as f:
    for line in f:
        tokens = line.strip().split()
        tokens = [int(x) for x in tokens]
        train_ent_pos.append(tokens)

test_ent_pos = []
with open('test_ent_pos.txt') as f:
    for line in f:
        tokens = line.strip().split()
        tokens = [int(x) for x in tokens]
        test_ent_pos.append(tokens)

print("word_vocab_size=%d\npos_vocab_size=%d\ndep_vocab_size=%d\nmax_len_path=%d"%(word_vocab_size, pos_vocab_size, dep_vocab_size, max_len_path))

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
desc_keep_prob=tf.placeholder(tf.float32, name="desc_keep_prob")
other = tf.placeholder(tf.bool, name="other")

with tf.name_scope("input"):
    path_length = tf.placeholder(tf.int32, shape=[None], name="path_length")
    word_ids = tf.placeholder(tf.int32, shape=[None, max_len_path], name="word_ids")
    pos_ids = tf.placeholder(tf.int32, [None, max_len_path], name="pos_ids")
    dep_ids = tf.placeholder(tf.int32, [None, max_len_path], name="dep_ids")
    dep_ids_reverse = tf.placeholder(tf.int32, [None, max_len_path], name="dep_ids")

    conv_mask=tf.placeholder(tf.float32, [None, max_len_path-1,1], "conv_mask")

    input_entity1_desc = tf.placeholder(tf.int32, [None, max_entity_desc_length], "input_entity1_desc")
    input_entity2_desc = tf.placeholder(tf.int32, [None, max_entity_desc_length], "input_entity2_desc")

    y1 = tf.placeholder(tf.int64, [None], name="y1")
    y2 = tf.placeholder(tf.int64, [None], name="y2")
    y = tf.placeholder(tf.int64, [None], name="y")

input_gate=tf.get_variable("input_gate",[1,convolution_state_size])
select_mask=tf.get_variable("select_mask",[1,relation_classes])
# tf.device("/cpu:0")
with tf.name_scope("word_embedding"):
    W = tf.Variable(tf.constant(0.0, shape=[word_vocab_size, word_embd_dim]), name="W", trainable=False)
    embedding_placeholder = tf.placeholder(tf.float32, [word_vocab_size, word_embd_dim])
    embedding_init = W.assign(embedding_placeholder)
    embedded_word = tf.nn.embedding_lookup(W, word_ids)
    word_embedding_saver = tf.train.Saver({"word_embedding/W": W})

############
#entity desc
with tf.variable_scope("desc_embedding"):

    en1_desc_em=tf.nn.embedding_lookup(W,input_entity1_desc)
    en2_desc_em = tf.nn.embedding_lookup(W, input_entity2_desc)

with tf.name_scope("pos_embedding"):
    W = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embd_dim], -0.1, 0.1), name="W")
    embedded_pos = tf.nn.embedding_lookup(W, pos_ids)
    pos_embedding_saver = tf.train.Saver({"pos_embedding/W": W})

if pos:
    embedded_word = tf.concat([embedded_word, embedded_pos], axis=2)

with tf.name_scope("dep_embedding"):
    W = tf.Variable(tf.random_uniform([dep_vocab_size, dep_embd_dim], -0.01, 0.01), name="W")
    embedded_dep = tf.nn.embedding_lookup(W, dep_ids)
    if directed:
        embedded_dep_reverse = tf.nn.embedding_lookup(W, dep_ids_reverse)
        embedded_dep_reverse_drop = tf.nn.dropout(embedded_dep_reverse, keep_prob)
    dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})

with tf.name_scope("dropout"):
    embedded_word_drop = tf.nn.dropout(embedded_word, keep_prob)
    embedded_dep_drop = tf.nn.dropout(embedded_dep, keep_prob)

    en1_desc_em = tf.nn.dropout(en1_desc_em, keep_prob)
    en2_desc_em = tf.nn.dropout(en2_desc_em, keep_prob)

en1_desc_em_4dim=tf.expand_dims(en1_desc_em,axis=-1)
en2_desc_em_4dim=tf.expand_dims(en2_desc_em,axis=-1)

word_hidden_state = tf.zeros([batch_size, word_state_size], name='word_hidden_state')
word_cell_state = tf.zeros([batch_size, word_state_size], name='word_cell_state')
word_init_state = tf.contrib.rnn.LSTMStateTuple(word_hidden_state, word_cell_state)



################
# attention related
################
he_normal = tf.keras.initializers.he_normal()
def slice_entity(inputs, ent_pos):
    '''
    Args
      conv_out: [batch, max_len, dim]
      ent_pos:  [batch, 4]
    '''
    # slice ent1
    # -------(e1.first--e1.last)-------e2.first--e2.last-------
    begin1 = ent_pos[:, 0]
    size1 = ent_pos[:, 1] - ent_pos[:, 0]

    # slice ent2
    # -------e1.first--e1.last-------(e2.first--e2.last)-------
    begin2 = ent_pos[:, 2]
    size2 = ent_pos[:, 3] - ent_pos[:, 2]
    
    entities = slice_batch_n(inputs, [begin1, begin2], [size1, size2])
    dim = inputs.shape.as_list()[-1]
    entities.set_shape(tf.TensorShape([None, None, dim]))

    return entities

def attention(inputs, name, reuse=None):
    H = inputs
    hidden_size = inputs.shape.as_list()[-1]
    with tf.variable_scope(name, reuse=reuse):
        M = tf.nn.tanh(H) # b,n,d
        w = tf.get_variable('w-att',[1, hidden_size], initializer=he_normal)
        batch_size = tf.shape(H)[0]
        alpha = tf.matmul(tf.tile(tf.expand_dims(w, 0), [batch_size, 1, 1]),
                        M, transpose_b=True)
        alpha = tf.nn.softmax(alpha) # b,1,n
        r = tf.matmul(alpha, H) # b, 1, d
        return tf.squeeze(r, axis=1)

def compute_logits(embedded_word_drop, 
                   embedded_dep_drop,
                   en1_desc_em_4dim,
                   en2_desc_em_4dim,
                   reuse=None):
    with tf.variable_scope("word_lstm1", reuse=reuse):
        cell = tf.contrib.rnn.BasicLSTMCell(word_state_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        state_series_word1, _ = tf.nn.dynamic_rnn(cell, embedded_word_drop, sequence_length=path_length, initial_state=None, dtype=tf.float32)


    with tf.variable_scope("word_lstm2", reuse=reuse):
        cell = tf.contrib.rnn.BasicLSTMCell(word_state_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        state_series_word2, _ = tf.nn.dynamic_rnn(cell, tf.reverse(embedded_word_drop, axis=[1]), sequence_length=path_length, initial_state=None, dtype=tf.float32)

    with tf.variable_scope("dep_lstm1", reuse=reuse):
        cell = tf.contrib.rnn.BasicLSTMCell(dep_state_size, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        state_series_dep1, _ = tf.nn.dynamic_rnn(cell, embedded_dep_drop, sequence_length=path_length-1, initial_state=None, dtype=tf.float32)

    with tf.variable_scope("dep_lstm2", reuse=reuse):
        cell = tf.contrib.rnn.BasicLSTMCell(dep_state_size, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        if directed:
            embedded_dep_drop = embedded_dep_reverse_drop
        state_series_dep2, _ = tf.nn.dynamic_rnn(cell, tf.reverse(embedded_dep_drop, axis=[1]), sequence_length=path_length-1, initial_state=None, dtype=tf.float32)

    # state_series_dep1 = tf.concat([state_series_dep1, tf.zeros([batch_size, 1, dep_state_size])], 1)
    # state_series_dep2 = tf.concat([state_series_dep2, tf.zeros([batch_size, 1, dep_state_size])], 1)

    state_series1 = tf.concat([state_series_word1, state_series_dep1], 2)
    state_series2 = tf.concat([state_series_word2, state_series_dep2], 2)

    # with tf.variable_scope('attention', reuse=reuse):
    #     # inputs = embedded_word_drop
    #     # ent_out_dim = inputs.shape.as_list()[-1]

    #     # entities = slice_entity(inputs, ent_pos)
    #     # scaled_entities = multihead_attention(entities, inputs, None, ent_out_dim, 
    #     #                             ent_out_dim, ent_out_dim, 13)
    #     # ent_out = tf.nn.relu(scaled_entities)
    #     # ent_out = tf.reduce_max(ent_out, axis=1)
    #     att1 = attention(state_series1, 'att1', reuse=reuse)
    #     att2 = attention(state_series2, 'att2', reuse=reuse)
    #     att_out_dim = word_state_size + dep_state_size

    state_series1 = tf.reshape(state_series1, [-1, max_len_path*int((win_size+1)/2), dep_state_size])
    state_series2 = tf.reshape(state_series2, [-1, max_len_path*int((win_size+1)/2), dep_state_size])

    state_series1_4dim = tf.expand_dims(state_series1, axis=-1)
    state_series2_4dim = tf.expand_dims(state_series2, axis=-1)
    # print(state_series1)

    with tf.variable_scope("CNN1", reuse=reuse):
        filter_shape = [win_size, dep_state_size, 1, convolution_state_size]
        # tf.contrib.xa
        # w = tf.Variable(tf.random_uniform(filter_shape, -0.01, 0.01), name="w")
        # b = tf.Variable(tf.constant(0.1, shape=[convolution_state_size]), name="b")
        w = tf.get_variable('w', filter_shape, initializer=he_normal)
        b = tf.get_variable('b', [convolution_state_size], initializer=he_normal)
        conv = tf.nn.conv2d(state_series1_4dim, w, strides=[1, int((win_size+1)/2), dep_state_size, 1], padding="VALID",name="conv")
        conv_afterrelu = tf.nn.relu(tf.nn.bias_add(conv, b), name="conv_afterrelu")
        conv_afterrelu_temp=tf.reshape(conv_afterrelu,[-1,max_len_path-1,convolution_state_size])
        conv_afterrelu=tf.reshape(conv_afterrelu_temp*conv_mask,[-1,max_len_path-1,1,convolution_state_size])
        pooled1 = tf.nn.max_pool(conv_afterrelu, ksize=[1, max_len_path-1, 1, 1],
                                strides=[1, max_len_path-1, 1, 1], padding="VALID", name="max_pool")
        pooled1_flat = tf.reshape(pooled1, [-1, convolution_state_size])

    with tf.variable_scope("CNN2", reuse=reuse):
        filter_shape = [win_size, dep_state_size, 1, convolution_state_size]
        # w = tf.Variable(tf.random_uniform(filter_shape, -0.01, 0.01), name="w")
        # b = tf.Variable(tf.constant(0.1, shape=[convolution_state_size]), name="b")
        w = tf.get_variable('w', filter_shape, initializer=he_normal)
        b = tf.get_variable('b', [convolution_state_size], initializer=he_normal)
        conv = tf.nn.conv2d(state_series2_4dim, w, strides=[1, (win_size+1)/2, dep_state_size, 1], padding="VALID",name="conv")
        conv_afterrelu = tf.nn.relu(tf.nn.bias_add(conv, b), name="conv_afterrelu")
        conv_afterrelu_temp=tf.reshape(conv_afterrelu,[-1,max_len_path-1,convolution_state_size])
        conv_afterrelu=tf.reshape(conv_afterrelu_temp*conv_mask,[-1,max_len_path-1,1,convolution_state_size])
        pooled2 = tf.nn.max_pool(conv_afterrelu, ksize=[1, max_len_path-1, 1, 1],
                                strides=[1, max_len_path-1, 1, 1], padding="VALID", name="max_pool")
        pooled2_flat = tf.reshape(pooled2, [-1, convolution_state_size])

    # with tf.name_scope("hidden_layer"):
    #     W = tf.Variable(tf.truncated_normal([convolution_state_size, 100], -0.1, 0.1), name="W")
    #     b = tf.Variable(tf.zeros([100]), name="b")
    #     y_hidden_layer = tf.matmul(pooled1_flat, W) + b

    with tf.name_scope("dropout"):
        pooled1_drop = tf.nn.dropout(pooled1_flat, keep_prob, name='pooled1_drop')
        pooled2_drop = tf.nn.dropout(pooled2_flat, keep_prob, name='pooled2_drop')


    #####################################################
    #entity description cnn
    ##################
    desc_l2_loss=tf.constant(0.0)
    with tf.variable_scope("DESC_CNN", reuse=reuse) as scope:
        # convolution layer
        filter_shape = [desc_filter_size,word_embd_dim , 1, desc_num_filters]
        desc_w = tf.get_variable(name="desc_w", shape=filter_shape, dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(True))
        desc_b = tf.get_variable(name="desc_b" , shape=[desc_num_filters],
                            initializer=tf.constant_initializer(0.1))
        #desc_l2_loss+=tf.nn.l2_loss(desc_w)
        #desc_l2_loss+=tf.nn.l2_loss(desc_b)

        conv_en1 = tf.nn.conv2d(en1_desc_em_4dim, desc_w, strides=[1, 1, word_embd_dim, 1], padding="SAME",
                            name="conv_en1")

        conv_en2 = tf.nn.conv2d(en2_desc_em_4dim, desc_w, strides=[1, 1, word_embd_dim, 1], padding="SAME",
                                name="conv_en2")
        # 对卷击结果进行Relu激活
        conv_en1_activation = tf.nn.relu(tf.nn.bias_add(conv_en1, desc_b), name="conv_en1_activation")

        conv_en2_activation = tf.nn.relu(tf.nn.bias_add(conv_en2, desc_b), name="conv_en2_activation")

        # max_pool 上面的输出
        desc1_pooled = tf.nn.max_pool(conv_en1_activation, ksize=[1, max_entity_desc_length, 1, 1],
                                strides=[1, max_entity_desc_length, 1, 1], padding="SAME", name="desc1_pooled")

        desc2_pooled = tf.nn.max_pool(conv_en2_activation, ksize=[1, max_entity_desc_length, 1, 1],
                                    strides=[1, max_entity_desc_length, 1, 1], padding="SAME", name="desc2_pooled")

        # batch norm
        # desc1_pooled = tf.layers.batch_normalization(desc1_pooled, training=is_train)
        # desc2_pooled = tf.layers.batch_normalization(desc2_pooled, training=is_train)

        desc1_pooled = tf.reshape(desc1_pooled, [-1, desc_num_filters])
        desc2_pooled = tf.reshape(desc2_pooled, [-1, desc_num_filters])

        with tf.variable_scope("desc_dropout"):
            desc1_pooled = tf.nn.dropout(desc1_pooled, desc_keep_prob)
            desc2_pooled = tf.nn.dropout(desc2_pooled, desc_keep_prob)

        with tf.variable_scope("desc_output") as scope:
            desc_features = tf.concat([desc1_pooled, desc2_pooled], axis=1)
            w_add = tf.get_variable(name="w_add", shape=[desc_num_filters * 2, relation_classes],
                                    initializer=tf.contrib.layers.xavier_initializer(True))
            b_add = tf.get_variable(name="b_add", shape=[relation_classes], initializer=tf.constant_initializer(0.1))
            desc_l2_loss += tf.nn.l2_loss(w_add)
            desc_l2_loss += tf.nn.l2_loss(b_add)
            desc_scores_add = tf.nn.xw_plus_b(desc_features, w_add, b_add, name="desc_scores_add")
            # desc_scores_add=tf.nn.dropout(tf.nn.relu(desc_scores_add),keep_prob)
            desc_scores_pro = tf.nn.softmax(desc_scores_add)

            # w_gate=tf.get_variable("w_gate",[convolution_state_size*2,convolution_state_size],initializer=tf.contrib.layers.xavier_initializer(True))
            # b_gate=tf.get_variable("b_gate",[convolution_state_size], initializer=tf.constant_initializer(0.1))
            # desc_l2_loss+=tf.nn.l2_loss(w_gate)
            # desc_l2_loss += tf.nn.l2_loss(b_gate)
            # gate_value=tf.sigmoid(tf.matmul(tf.concat([pooled1_drop,desc_scores_add],axis=1),w_gate)+b_gate)

            # gate_value=tf.sigmoid(input_gate)
            # pooled1_drop=pooled1_drop+gate_value*desc_scores_add
            #pooled2_drop=pooled2_drop+gate_value*desc_scores_add
    with tf.variable_scope("softmax_layer1", reuse=reuse):
        # W = tf.Variable(tf.random_uniform([convolution_state_size, relation_classes], -0.1, 0.1), name="W")
        # b = tf.Variable(tf.zeros([relation_classes]), name="b")
        W = tf.get_variable('w', [convolution_state_size, relation_classes], 
                                 initializer=he_normal)
        b = tf.get_variable('b', [relation_classes], initializer=he_normal)
        logits1 = tf.matmul(pooled1_drop, W) + b
        predictions1 = tf.argmax(logits1, 1)

    with tf.name_scope("softmax_layer2"):
        # W = tf.Variable(tf.random_uniform([convolution_state_size, relation_classes], -0.1, 0.1), name="W")
        # b = tf.Variable(tf.zeros([relation_classes]), name="b")
        logits2 = tf.matmul(pooled2_drop, W) + b
        predictions2 = tf.argmax(logits2, 1)

    with tf.variable_scope("softmax_layer", reuse=reuse):
        pooled_drop =  tf.cond(other, lambda: tf.concat([pooled1_drop, tf.zeros_like(pooled2_drop)], 1), lambda: tf.concat([pooled1_drop, pooled2_drop] ,1))
        # pooled_drop = tf.concat([pooled1_drop, pooled2_drop], 1)
        pooled_drop = tf.reshape(pooled_drop, [-1, convolution_state_size*2])
        # W = tf.Variable(tf.random_uniform([convolution_state_size*2, 10], -0.1, 0.1), name="W")
        # b = tf.Variable(tf.zeros([10]), name="b")
        W = tf.get_variable('w', [convolution_state_size*2, 10], 
                                 initializer=he_normal)
        b = tf.get_variable('b', [10], initializer=he_normal)
        logits = tf.matmul(pooled_drop, W) + b
        predictions = tf.argmax(logits, 1)

    predictions_test = tf.argmax((1-belda)*(alpha*tf.nn.softmax(logits1) + (1-alpha)*tf.nn.softmax(logits2[::-1]))+belda*desc_scores_pro, 1)

    return (logits1, logits2, logits), \
           (predictions1, predictions2, predictions, predictions_test), \
           desc_l2_loss, desc_scores_pro

(logits1, logits2, logits), \
    (predictions1, predictions2, predictions, predictions_test), \
    desc_l2_loss, desc_scores_pro = compute_logits(
                   embedded_word_drop, 
                   embedded_dep_drop,
                   en1_desc_em_4dim,
                   en2_desc_em_4dim,
                   reuse=tf.AUTO_REUSE)

tv_all = tf.trainable_variables()
tv_regu = []
non_reg = ["word_embedding/W:0","pos_embedding/W:0",'dep_embedding/W:0',
           "global_step:0","input_gate:0","DESC_CNN/desc_output/w_add:0",
           "DESC_CNN/desc_output/b_add:0", "DESC_CNN/desc_w:0",
           "DESC_CNN/desc_b:0"]
for t in tv_all:
    if t.name not in non_reg:
        if(t.name.find('biases')==-1):
            tv_regu.append(t)
# print(tv_regu)

with tf.name_scope("loss"):
    l2_loss = lambda_l2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])
    #################
    # l2 loss desc  use different lambda_l2 later
    l2_loss += desc_lambda_l2*desc_l2_loss

def compute_xentropy_loss(logits1, logits2, logits, desc_scores_pro):
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=y1))
        loss += tf.cond(other, lambda: 0.0, lambda: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=y2)))
        # W = tf.Variable(tf.random_uniform([convolution_state_size, 10], -0.1, 0.1), name="W")
        # b = tf.Variable(tf.zeros([10]), name="b")
        # logits_coarse = tf.matmul(pooled1_drop, W) + b
        # loss += tf.cond(other, lambda: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_coarse, labels=y)), lambda: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)))
        loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        # DESC PART  loss
        loss+=tf.reduce_mean(tf.reduce_sum(-tf.one_hot(y1,relation_classes)*tf.log(desc_scores_pro),axis=1) )
        #tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=desc_scores_add, labels=y1))
        
        return loss


loss_xent = compute_xentropy_loss(logits1, logits2, logits, desc_scores_pro)


################
# adv related
################
def adv_example(inputs, loss):
    grad, = tf.gradients(
        loss,
        inputs,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    perturb = scale_l2(grad)

    return inputs + perturb

def scale_l2(x, eps=1e-3):
    # scale over the full batch
    return eps * tf.nn.l2_normalize(x, dim=[0, 1, 2])

adv_word = adv_example(embedded_word_drop, loss_xent)
adv_dep = adv_example(embedded_dep_drop, loss_xent)
adv_en1_desc = adv_example(en1_desc_em_4dim, loss_xent)
adv_en2_desc = adv_example(en2_desc_em_4dim, loss_xent)

(logits1, logits2, logits), _, \
    desc_l2_loss, desc_scores_pro = compute_logits(
                   adv_word,#embedded_word_drop, 
                   adv_dep,#embedded_dep_drop,
                   adv_en1_desc,#en1_desc_em_4dim,
                   adv_en2_desc,#en2_desc_em_4dim,
                   reuse=tf.AUTO_REUSE)
adv_loss = compute_xentropy_loss(logits1, logits2, logits, desc_scores_pro)

total_loss = loss_xent + l2_loss + adv_loss #


with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions_test, y1)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
sess.run(embedding_init, feed_dict={embedding_placeholder:embeddings})

# sess.run(embedding_init, feed_dict={embedding_placeholder:word_embedding})
# word_embedding_saver.save(sess, word_embd_dir + '/word_embd')

# pos_embedding_saver.save(sess, pos_embd_dir + '/pos_embd')
# dep_embedding_saver.save(sess, dep_embd_dir + '/dep_embd')

# model = tf.train.latest_checkpoint(model_dir)
# saver.restore(sess, model)

# latest_embd = tf.train.latest_checkpoint(word_embd_dir)
# word_embedding_saver.restore(sess, latest_embd)

max_acc=0.
max_epoch=0
max_f1=0.0
for i in range(num_epochs):
    loss_per_epoch = 0
    acc_per_epoch = 0
    if shuffle:
        data_zip = list(zip(word_p_ids,pos_p_ids,dep_p_ids,dep_p_ids_reverse,rel_ids,path_len,train_en1_to_desc_id,train_en2_to_desc_id))
        data_zip = np.asarray(data_zip)
        shuffle_idx = np.random.permutation(np.arange(length))
        shuffled_data = data_zip[shuffle_idx]
        _, __, ___, ____, rel_id, _____,______,_______ = zip(*shuffled_data)
        partitions = np.equal(rel_id, np.full(length,9))
        shuffled_data_part1 = shuffled_data[~partitions]
        shuffled_data_part2 = shuffled_data[partitions]
        word_p_ids_part1,pos_p_ids_part1,dep_p_ids_part1,dep_p_ids_reverse_part1,rel_ids_part1,path_len_part1,en1_to_desc_id_part1,en2_to_desc_id_part1 = zip(*shuffled_data_part1)
        word_p_ids_part2,pos_p_ids_part2,dep_p_ids_part2,dep_p_ids_reverse_part2,rel_ids_part2,path_len_part2 ,en1_to_desc_id_part2,en2_to_desc_id_part2= zip(*shuffled_data_part2)

    num_batches_part1 = int(math.ceil(len(shuffled_data_part1)/BATCH_SIZE))
    # print(len(shuffled_data_part1),num_batches_part1)
    num_batches_part2 = int(math.ceil(len(shuffled_data_part2)/BATCH_SIZE))
    # print(len(shuffled_data_part2),num_batches_part2)
    all_predictions = []
    for j in range(num_batches_part1):
        start_index = j*BATCH_SIZE
        end_index = min((j+1)*BATCH_SIZE, len(shuffled_data_part1))
        path_dict = path_len_part1[start_index:end_index]
        word_dict = word_p_ids_part1[start_index:end_index]
        pos_dict = pos_p_ids_part1[start_index:end_index]
        dep_dict = dep_p_ids_part1[start_index:end_index]
        dep_dict_reverse = dep_p_ids_reverse_part1[start_index:end_index]

        entity1_desc = en1_to_desc_id_part1[start_index:end_index]
        entity2_desc = en2_to_desc_id_part1[start_index:end_index]

        y1_dict = rel_ids_part1[start_index:end_index]
        y2_dict = [-(y1+1-19) for y1 in y1_dict]
        y_dict = [-(y1+1-19) if y1>9 else y1 for y1 in y1_dict]

        conv_mask_feed=np.zeros([len(word_dict),max_len_path-1,1],np.float32)
        for j1,p1 in enumerate(path_dict):
            for i1 in range(p1-1):
                conv_mask_feed[j1][i1][0]=1.0

        feed_dict = {
            path_length:path_dict,
            word_ids:word_dict,
            pos_ids:pos_dict,
            dep_ids:dep_dict,
            dep_ids_reverse:dep_dict_reverse,
            conv_mask:conv_mask_feed,
            input_entity1_desc: entity1_desc,
            input_entity2_desc: entity2_desc,
            y1:y1_dict,
            y2:y2_dict,
            y:y_dict,
            keep_prob:dropout_keep_prob,
            desc_keep_prob: dropout_desc_keep,
            other:False}
        _, _loss, step, batch_predictions = sess.run([optimizer, total_loss, global_step, predictions_test], feed_dict)
        all_predictions.append(batch_predictions)
        loss_per_epoch += _loss

    for j in range(num_batches_part2):
        start_index = j*BATCH_SIZE
        end_index = min((j+1)*BATCH_SIZE, len(shuffled_data_part2))
        path_dict = path_len_part2[start_index:end_index]
        word_dict = word_p_ids_part2[start_index:end_index]
        pos_dict = pos_p_ids_part2[start_index:end_index]
        dep_dict = dep_p_ids_part2[start_index:end_index]
        dep_dict_reverse = dep_p_ids_reverse_part2[start_index:end_index]

        entity1_desc = en1_to_desc_id_part2[start_index:end_index]
        entity2_desc = en2_to_desc_id_part2[start_index:end_index]

        y1_dict = rel_ids_part2[start_index:end_index]
        y2_dict = [-(y1+1-19) for y1 in y1_dict]
        y_dict = [-(y1+1-19) if y1>9 else y1 for y1 in y1_dict]

        conv_mask_feed=np.zeros([len(word_dict),max_len_path-1,1],np.float32)
        for j1,p1 in enumerate(path_dict):
            for i1 in range(p1-1):
                conv_mask_feed[j1][i1][0]=1.0

        feed_dict = {
            path_length:path_dict,
            word_ids:word_dict,
            pos_ids:pos_dict,
            dep_ids:dep_dict,
            dep_ids_reverse:dep_dict_reverse,
            conv_mask:conv_mask_feed,
            input_entity1_desc: entity1_desc,
            input_entity2_desc: entity2_desc,
            y1:y1_dict,
            y2:y2_dict,
            y:y_dict,
            keep_prob:dropout_keep_prob,
            desc_keep_prob: dropout_desc_keep,
            other:True}
        _, _loss, step, batch_predictions = sess.run([optimizer, total_loss, global_step, predictions_test], feed_dict)
        all_predictions.append(batch_predictions)
        loss_per_epoch += _loss
    # training accuracy
    y_pred = []
    for j in range(num_batches_part1+num_batches_part2):
        for pred in all_predictions[j]:
            y_pred.append(pred)
    count = 0
    rel_ids_shuffled = np.concatenate((rel_ids_part1, rel_ids_part2))
    for j in range(length):
        count += y_pred[j] == rel_ids_shuffled[j]
    accuracy = 1.0*count/length * 100
    time_str = datetime.datetime.now().isoformat()
    print(time_str, "Epoch:", i+1, "Step:", step, "loss:", loss_per_epoch/(num_batches_part1+num_batches_part2), "train accuracy:", accuracy)

    # test predictions
    all_predictions = []
    for j in range(num_batches_test):
        end_index = min((j+1)*BATCH_SIZE, length_test)
        path_dict = path_len_test[j*batch_size:end_index]
        word_dict = word_p_ids_test[j*batch_size:end_index]
        pos_dict = pos_p_ids_test[j*batch_size:end_index]
        dep_dict = dep_p_ids_test[j*batch_size:end_index]
        dep_dict_reverse = dep_p_ids_test_reverse[j*batch_size:end_index]

        entity1_desc = test_en1_to_desc_id[j * batch_size:end_index]
        entity2_desc = test_en2_to_desc_id[j * batch_size:end_index]

        y1_dict = rel_ids_test[j*batch_size:end_index]
        y2_dict = [-(y1+1-19) for y1 in y1_dict]
        y_dict = [-(y1+1-19) if y1>9 else y1 for y1 in y1_dict]

        test_conv_mask_feed = np.zeros([len(word_dict), max_len_path - 1, 1], np.float32)
        for j1, p1 in enumerate(path_dict):
            for i1 in range(p1 - 1):
                test_conv_mask_feed[j1][i1][0] = 1.0

        feed_dict = {
            path_length:path_dict,
            word_ids:word_dict,
            pos_ids:pos_dict,
            dep_ids:dep_dict,
            dep_ids_reverse:dep_dict_reverse,
            conv_mask:test_conv_mask_feed,
            input_entity1_desc: entity1_desc,
            input_entity2_desc: entity2_desc,
            y1:y1_dict,
            y2:y2_dict,
            y:y_dict,
            keep_prob:1.0,
            desc_keep_prob: 1.0
    }
        batch_predictions = sess.run(predictions_test, feed_dict)
        all_predictions.append(batch_predictions)

    y_pred = []
    for j in range(num_batches_test):
        for pred in all_predictions[j]:
            y_pred.append(pred)

    count = 0
    for j in range(length_test):
        count += y_pred[j] == rel_ids_test[j]
    accuracy = 1.0 * count / length_test * 100

    prediction_result_file = open(data_dir + '/prediction_result.txt', 'w')
    real_result_file = open(data_dir + "/real_result.txt", 'w')
    result_scores_file = open(data_dir + '/result_scores.txt', 'w')
    for j in range(length_test):
        real_result_file.write(str(j) + '\t' + id2rel[rel_ids_test[j]] + '\n')
        prediction_result_file.write(str(j) + '\t' + id2rel[y_pred[j]] + '\n')
    prediction_result_file.close()
    real_result_file.close()
    output = subprocess.getoutput(
        'perl data/semeval2010_task8_scorer-v1.2.pl data/prediction_result.txt data/real_result.txt')
    f1 = float(output[-10:-5])
    print('test accracy', accuracy, 'f1', f1)
    if f1 > max_f1:
        max_f1 = f1
        max_acc = accuracy
        max_epoch = i
        result_scores_file.write(output)
        result_scores_file.close()

    # f1 = f1_score(rel_ids_test[:2700], y_pred, average='macro')
    # print("sklearn f1_score:", f1)
    print('')

print("epoch:", max_epoch + 1, "accuracy:", max_acc, 'max_f1:', max_f1)
saver.save(sess, model_dir)
print("Saved Model")

###########################
#add desc part model need modify  4 places
