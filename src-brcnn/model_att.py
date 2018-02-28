import sys, os, _pickle as pickle
import tensorflow as tf
import numpy as np
#from sklearn.metrics import f1_score
from config import *
from attention import *
import gensim
import datetime
import math

embeddings = []
word2id = {}
id2word = {}
pad_word = "<pad>"
word2id[pad_word] = 0
id2word[0] = pad_word
embeddings.append([0 for i in range(word_embd_dim)])

if not senna:
    f = open(data_dir + '/essential_vectors-3.pkl', 'rb')
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
        # w = w.lower()
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

embeddings = np.asarray(embeddings)
miss_embe_count = len(word2id)-init_vocab_count
miss_embeddings = np.zeros([miss_embe_count, word_embd_dim], dtype=float)
embeddings = np.vstack((embeddings, miss_embeddings))

relations_test = []
for line in open(data_dir + '/test_relations.txt'):
    relations_test.append(line.strip().split()[0])

length_test = len(word_p_test)
num_batches_test = int(math.ceil(length_test/BATCH_SIZE))

for i in range(length_test):
    for j, word in enumerate(word_p_test[i]):
        # word = word.lower()
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

batch_size = BATCH_SIZE
word_vocab_size = len(word2id)
pos_vocab_size = len(pos_tag2id)
dep_vocab_size = len(dep2id)
print("word_vocab_size=%d\npos_vocab_size=%d\ndep_vocab_size=%d\nmax_len_path=%d"%(word_vocab_size, pos_vocab_size, dep_vocab_size, max_len_path))

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

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
other = tf.placeholder(tf.bool, name="other")

with tf.name_scope("input"):
    path_length = tf.placeholder(tf.int32, shape=[None], name="path_length")
    word_ids = tf.placeholder(tf.int32, shape=[None, max_len_path], name="word_ids")
    pos_ids = tf.placeholder(tf.int32, [None, max_len_path], name="pos_ids")
    dep_ids = tf.placeholder(tf.int32, [None, max_len_path], name="dep_ids")
    dep_ids_reverse = tf.placeholder(tf.int32, [None, max_len_path], name="dep_ids")
    y1 = tf.placeholder(tf.int64, [None], name="y1")
    y2 = tf.placeholder(tf.int64, [None], name="y2")
    y = tf.placeholder(tf.int64, [None], name="y")

    ent_pos = tf.placeholder(tf.int32, [None, 4], name="ent_pos")

# tf.device("/cpu:0")
with tf.name_scope("word_embedding"):
    W = tf.Variable(tf.constant(0.0, shape=[word_vocab_size, word_embd_dim]), name="W", trainable=False)
    embedding_placeholder = tf.placeholder(tf.float32, [word_vocab_size, word_embd_dim])
    embedding_init = W.assign(embedding_placeholder)
    embedded_word = tf.nn.embedding_lookup(W, word_ids)
    word_embedding_saver = tf.train.Saver({"word_embedding/W": W})

with tf.name_scope("pos_embedding"):
    W = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embd_dim], -0.1, 0.1), name="W")
    embedded_pos = tf.nn.embedding_lookup(W, pos_ids)
    pos_embedding_saver = tf.train.Saver({"pos_embedding/W": W})

if pos:
    embedded_word = tf.concat([embedded_word, embedded_pos], axis=2)

with tf.name_scope("dep_embedding"):
    W = tf.Variable(tf.random_uniform([dep_vocab_size, dep_embd_dim], -0.1, 0.1), name="W")
    embedded_dep = tf.nn.embedding_lookup(W, dep_ids)
    if directed:
        embedded_dep_reverse = tf.nn.embedding_lookup(W, dep_ids_reverse)
        embedded_dep_reverse_drop = tf.nn.dropout(embedded_dep_reverse, keep_prob)
    dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})

with tf.name_scope("dropout"):
    embedded_word_drop = tf.nn.dropout(embedded_word, keep_prob)
    embedded_dep_drop = tf.nn.dropout(embedded_dep, keep_prob)

# word_hidden_state = tf.zeros([batch_size, word_state_size], name='word_hidden_state')
# word_cell_state = tf.zeros([batch_size, word_state_size], name='word_cell_state')
# word_init_state = tf.contrib.rnn.LSTMStateTuple(word_hidden_state, word_cell_state)

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
                   reuse=None):
    with tf.variable_scope("word_lstm1", reuse=reuse):
        cell = tf.contrib.rnn.BasicLSTMCell(word_state_size)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        state_series_word1, _ = tf.nn.dynamic_rnn(
            cell, embedded_word_drop, sequence_length=path_length, 
            initial_state=None, dtype=tf.float32)

    with tf.variable_scope("word_lstm2", reuse=reuse):
        cell = tf.contrib.rnn.BasicLSTMCell(word_state_size)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        state_series_word2, _ = tf.nn.dynamic_rnn(
            cell, tf.reverse(embedded_word_drop, axis=[1]), 
            sequence_length=path_length, initial_state=None, dtype=tf.float32)

    with tf.variable_scope("dep_lstm1", reuse=reuse):
        cell = tf.contrib.rnn.BasicLSTMCell(dep_state_size, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        state_series_dep1, _ = tf.nn.dynamic_rnn(
            cell, embedded_dep_drop, sequence_length=path_length-1, 
            initial_state=None, dtype=tf.float32)

    with tf.variable_scope("dep_lstm2", reuse=reuse):
        cell = tf.contrib.rnn.BasicLSTMCell(dep_state_size, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        if directed:
            embedded_dep_drop = embedded_dep_reverse_drop
        state_series_dep2, _ = tf.nn.dynamic_rnn(
            cell, tf.reverse(embedded_dep_drop, axis=[1]), 
            sequence_length=path_length-1, initial_state=None, dtype=tf.float32)

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

    state_series1 = tf.reshape(state_series1, 
            [-1, max_len_path*int((win_size+1)/2), dep_state_size])
    state_series2 = tf.reshape(state_series2, 
            [-1, max_len_path*int((win_size+1)/2), dep_state_size])

    state_series1_4dim = tf.expand_dims(state_series1, axis=-1)
    state_series2_4dim = tf.expand_dims(state_series2, axis=-1)
    # print(state_series1)

    with tf.variable_scope("CNN1", reuse=reuse):
        filter_shape = [win_size, dep_state_size, 1, convolution_state_size]
        # tf.contrib.xa
        w = tf.get_variable('w', filter_shape, initializer=he_normal)
        b = tf.get_variable('b', [convolution_state_size], initializer=he_normal)
        conv = tf.nn.conv2d(state_series1_4dim, w, strides=[1, int((win_size+1)/2), dep_state_size, 1], padding="VALID",
                            name="conv")
        conv_afterrelu = tf.nn.relu(tf.nn.bias_add(conv, b), name="conv_afterrelu")
        pooled1 = tf.nn.max_pool(conv_afterrelu, ksize=[1, max_len_path-1, 1, 1],
                                strides=[1, max_len_path-1, 1, 1], padding="SAME", name="max_pool")
        pooled1_flat = tf.reshape(pooled1, [-1, convolution_state_size])

    with tf.variable_scope("CNN2", reuse=reuse):
        filter_shape = [win_size, dep_state_size, 1, convolution_state_size]

        w = tf.get_variable('w', filter_shape, initializer=he_normal)
        b = tf.get_variable('b', [convolution_state_size], initializer=he_normal)
        conv = tf.nn.conv2d(state_series2_4dim, w, strides=[1, (win_size+1)/2, dep_state_size, 1], padding="VALID",
                            name="conv")
        conv_afterrelu = tf.nn.relu(tf.nn.bias_add(conv, b), name="conv_afterrelu")
        pooled2 = tf.nn.max_pool(conv_afterrelu, ksize=[1, max_len_path-1, 1, 1],
                                strides=[1, max_len_path-1, 1, 1], padding="SAME", name="max_pool")
        pooled2_flat = tf.reshape(pooled2, [-1, convolution_state_size])

    with tf.name_scope("dropout"):
        pooled1_drop = tf.nn.dropout(pooled1_flat, keep_prob, name='pooled1_drop')
        pooled2_drop = tf.nn.dropout(pooled2_flat, keep_prob, name='pooled2_drop')
        # att1_drop = tf.nn.dropout(att1, keep_prob)
        # att2_drop = tf.nn.dropout(att2, keep_prob)

    with tf.variable_scope("softmax_layer1", reuse=reuse):
        W = tf.get_variable('w', [convolution_state_size, relation_classes], 
                                 initializer=he_normal)
        b = tf.get_variable('b', [relation_classes], initializer=he_normal)
        logits1 = tf.matmul(pooled1_drop, W) + b
        # W = tf.get_variable('w', [convolution_state_size+att_out_dim, relation_classes], 
        #                          initializer=he_normal)
        # b = tf.get_variable('b', [relation_classes], initializer=he_normal)
        # logits1 = tf.matmul(tf.concat([pooled1_drop, att1_drop], axis=1), 
        #                     W) + b
        predictions1 = tf.argmax(logits1, 1)

    with tf.name_scope("softmax_layer2"):
        logits2 = tf.matmul(pooled2_drop, W) + b
        # logits2 = tf.matmul(tf.concat([pooled2_drop, att2_drop], axis=1), 
        #                     W) + b
        predictions2 = tf.argmax(logits2, 1)

    with tf.variable_scope("softmax_layer", reuse=reuse):
        pooled_drop = tf.concat([pooled1_drop, pooled2_drop], 1)
        pooled_drop = tf.reshape(pooled_drop, [-1, convolution_state_size*2])
        W = tf.get_variable('w', [convolution_state_size*2, 10], 
                                 initializer=he_normal)
        b = tf.get_variable('b', [10], initializer=he_normal)
        logits = tf.matmul(pooled_drop, W) + b
        predictions = tf.argmax(logits, 1)

        # pooled_drop = tf.concat([pooled1_drop, pooled2_drop, ent_out_drop], 1)
        # pooled_drop = tf.reshape(pooled_drop, [-1, convolution_state_size*2+ent_out_dim])
        # W = tf.get_variable('w', [convolution_state_size*2+ent_out_dim, 10], 
        #                          initializer=he_normal)
        # b = tf.get_variable('b', [10], initializer=he_normal)
        # logits = tf.matmul(pooled_drop, W) + b
        # predictions = tf.argmax(logits, 1)

    predictions_test = tf.argmax(alpha*tf.nn.softmax(logits1) + (1-alpha)*tf.nn.softmax(logits2[::-1]), 1)

    return (logits1, logits2, logits), \
           (predictions1, predictions2, predictions, predictions_test)

(logits1, logits2, logits), \
    (predictions1, predictions2, predictions, predictions_test) = compute_logits(
                   embedded_word_drop, 
                   embedded_dep_drop,
                   reuse=tf.AUTO_REUSE)

tv_all = tf.trainable_variables()
for tensor in tv_all:
    print(tensor.name)
# exit()

tv_regu = []
loss = 0.0
non_reg = ["word_embedding/W:0","pos_embedding/W:0",'dep_embedding/W:0',"global_step:0"]
for t in tv_all:
    if t.name not in non_reg:
        if 'multihead_attention' in t.name:
            continue
        if(t.name.find('biases')==-1):
            tv_regu.append(t)
l2_loss = lambda_l2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv_regu])

def compute_xentropy_loss(logits1, logits2, logits):
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=y1))
        loss += tf.cond(other, 
                     lambda: 0.0, 
                     lambda: tf.reduce_mean(
                       tf.nn.sparse_softmax_cross_entropy_with_logits(
                           logits=logits2, labels=y2)))
        
        loss += tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=y))
    return loss

loss_xent = compute_xentropy_loss(logits1, logits2, logits)

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
# adv_dep = adv_example(embedded_dep_drop, loss_xent)

(logits1, logits2, logits), _ = compute_logits(
                   adv_word, 
                   embedded_dep_drop,
                   reuse=tf.AUTO_REUSE)
adv_loss = compute_xentropy_loss(logits1, logits2, logits)

total_loss = loss_xent + l2_loss #+ adv_loss #

with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions_test, y)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
sess.run(embedding_init, feed_dict={embedding_placeholder:embeddings})

max_acc=0.
max_epoch=0
for i in range(num_epochs):
    loss_per_epoch = 0
    acc_per_epoch = 0
    if shuffle:
        data_zip = list(zip(train_ent_pos, word_p_ids,pos_p_ids,dep_p_ids,dep_p_ids_reverse,rel_ids,path_len))
        data_zip = np.asarray(data_zip)
        shuffle_idx = np.random.permutation(np.arange(length))
        shuffled_data = data_zip[shuffle_idx]
        partitions = np.equal(rel_ids, np.full(length,9))
        shuffled_data_part1 = shuffled_data[~partitions]
        shuffled_data_part2 = shuffled_data[partitions]
        train_ent_pos_part1, word_p_ids_part1,pos_p_ids_part1,dep_p_ids_part1,dep_p_ids_reverse_part1,rel_ids_part1,path_len_part1 = zip(*shuffled_data_part1)
        train_ent_pos_part2, word_p_ids_part2,pos_p_ids_part2,dep_p_ids_part2,dep_p_ids_reverse_part2,rel_ids_part2,path_len_part2 = zip(*shuffled_data_part2)

    num_batches_part1 = int(math.ceil(len(shuffled_data_part1)/BATCH_SIZE))
    # print(len(shuffled_data_part1),num_batches_part1)
    num_batches_part2 = int(math.ceil(len(shuffled_data_part2)/BATCH_SIZE))
    # print(len(shuffled_data_part2),num_batches_part2)
    all_predictions = []
    for j in range(num_batches_part1):
        start_index = j*BATCH_SIZE
        end_index = min((j+1)*BATCH_SIZE, len(shuffled_data_part1))
        path_dict = path_len_part1[start_index:end_index]
        ent_pos_dict = train_ent_pos_part1[start_index:end_index]
        word_dict = word_p_ids_part1[start_index:end_index]
        pos_dict = pos_p_ids_part1[start_index:end_index]
        dep_dict = dep_p_ids_part1[start_index:end_index]
        dep_dict_reverse = dep_p_ids_reverse_part1[start_index:end_index]
        y1_dict = rel_ids_part1[start_index:end_index]
        y2_dict = [-(y1+1-19) for y1 in y1_dict]
        y_dict = [-(y1+1-19) if y1>9 else y1 for y1 in y1_dict]
        

        feed_dict = {
            path_length:path_dict,
            ent_pos: ent_pos_dict,
            word_ids:word_dict,
            pos_ids:pos_dict,
            dep_ids:dep_dict,
            dep_ids_reverse:dep_dict_reverse,
            y1:y1_dict,
            y2:y2_dict,
            y:y_dict,
            keep_prob:dropout_keep_prob,
            other:False}
        _, _loss, step, batch_predictions = sess.run([optimizer, total_loss, global_step, predictions_test], feed_dict)
        all_predictions.append(batch_predictions)
        loss_per_epoch += _loss

    for j in range(num_batches_part2):
        start_index = j*BATCH_SIZE
        end_index = min((j+1)*BATCH_SIZE, len(shuffled_data_part2))
        path_dict = path_len_part2[start_index:end_index]
        ent_pos_dict = train_ent_pos_part2[start_index:end_index]
        word_dict = word_p_ids_part2[start_index:end_index]
        pos_dict = pos_p_ids_part2[start_index:end_index]
        dep_dict = dep_p_ids_part2[start_index:end_index]
        dep_dict_reverse = dep_p_ids_reverse_part2[start_index:end_index]
        y1_dict = rel_ids_part2[start_index:end_index]
        y2_dict = [-(y1+1-19) for y1 in y1_dict]
        y_dict = [-(y1+1-19) if y1>9 else y1 for y1 in y1_dict]

        feed_dict = {
            path_length:path_dict,
            ent_pos: ent_pos_dict,
            word_ids:word_dict,
            pos_ids:pos_dict,
            dep_ids:dep_dict,
            dep_ids_reverse:dep_dict_reverse,
            y1:y1_dict,
            y2:y2_dict,
            y:y_dict,
            keep_prob:dropout_keep_prob,
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
    # time_str = datetime.datetime.now().isoformat()
    # print(time_str, "Epoch:", i+1, "Step:", step, "loss:", loss_per_epoch/(num_batches_part1+num_batches_part2), "train accuracy:", accuracy)
    train_msg = "Epoch %d Step %d loss_acc %.2f %.2f" % (i+1, step, loss_per_epoch/(num_batches_part1+num_batches_part2), accuracy)

    # test predictions
    all_predictions = []
    for j in range(num_batches_test):
        end_index = min((j+1)*BATCH_SIZE, length_test)
        path_dict = path_len_test[j*batch_size:end_index]
        ent_pos_dict = test_ent_pos[j*batch_size:end_index]
        word_dict = word_p_ids_test[j*batch_size:end_index]
        pos_dict = pos_p_ids_test[j*batch_size:end_index]
        dep_dict = dep_p_ids_test[j*batch_size:end_index]
        dep_dict_reverse = dep_p_ids_test_reverse[j*batch_size:end_index]
        y1_dict = rel_ids_test[j*batch_size:end_index]
        y2_dict = [-(y1+1-19) for y1 in y1_dict]
        y_dict = [-(y1+1-19) if y1>9 else y1 for y1 in y1_dict]

        feed_dict = {
            path_length:path_dict,
            ent_pos: ent_pos_dict,
            word_ids:word_dict,
            pos_ids:pos_dict,
            dep_ids:dep_dict,
            dep_ids_reverse:dep_dict_reverse,
            y1:y1_dict,
            y2:y2_dict,
            y:y_dict,
            keep_prob:1.0}
        batch_predictions = sess.run(predictions_test, feed_dict)
        all_predictions.append(batch_predictions)

    y_pred = []
    for j in range(num_batches_test):
        for pred in all_predictions[j]:
            y_pred.append(pred)

    count = 0
    for j in range(length_test):
        count += y_pred[j] == rel_ids_test[j]
    accuracy = 1.0*count/length_test * 100

    print("%s %.2f" % (train_msg, accuracy))
    if accuracy > max_acc:
        max_acc = accuracy
        max_epoch = i
        prediction_result_file=open(data_dir + '/prediction_result.txt','w')
        real_result_file=open(data_dir + "/real_result.txt",'w')
        for j in range(length_test):
            real_result_file.write(str(j)+'\t'+id2rel[rel_ids_test[j]]+'\n')
            prediction_result_file.write(str(j)+'\t'+id2rel[y_pred[j]]+'\n')
        prediction_result_file.close()
        real_result_file.close()

    # f1 = f1_score(rel_ids_test[:2700], y_pred, average='macro')
    # print("sklearn f1_score:", f1)
    # print('')

print("epoch:", max_epoch+1, "max_accuracy:", max_acc)
saver.save(sess, model_dir)
print("Saved Model")
