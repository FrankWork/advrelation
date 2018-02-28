#coding:utf-8

import numpy as np
import re
import gensim


def parse_and_load_embedding(word_embedding_dim,pad_word,entity2id,big_dim_word_embed_path,use_big_dim_word_embed=False):
    #assert word_embedding_dim==50
    wlist_file = "./data/embedding/senna/words.lst"
    em_file = "./data/embedding/senna/embeddings.txt"

    embeddings=[]
    vecab={}

    if not use_big_dim_word_embed:
        wordlist=open(wlist_file,"r").readlines()
        allword_embedding=open(em_file,"r").readlines()
        for wid in range(len(wordlist)):
            word=wordlist[wid].strip()
            one_embedding=map(eval,allword_embedding[wid].strip().split())
            if(len(one_embedding)!=word_embedding_dim):
                print ("Error ,parse the "+wid +"id embedding error")
            else:
                embeddings.append(one_embedding)
                vecab[word]=wid
        for k,v in entity2id.items():
            if not vecab.has_key(k):

                num_word=k.split('_')
                em=np.zeros([word_embedding_dim],dtype=np.float32)
                for j in num_word:
                    if vecab.has_key(j):
                        em+=embeddings[vecab[j]]
                    else:
                        em+=np.random.normal(0.0,0.1,size=[word_embedding_dim])
                em=em/len(num_word)
                embeddings.append(em)
                vecab[k] = len(vecab)



    else:
        word_embedding_file = open(big_dim_word_embed_path, "r")
        for line in word_embedding_file:
            line = line.strip().strip("\n").split()
            vecab[line[0]] = len(vecab)
            one_embedding = map(eval, line[1:])
            embeddings.append(one_embedding)
        word_embedding_file.close()

        for k, v in entity2id.items():
            if not vecab.has_key(k):
                num_word = k.split('_')
                em = np.zeros([word_embedding_dim], dtype=np.float32)
                for j in num_word:
                    if vecab.has_key(j):
                        em += embeddings[vecab[j]]
                    else:
                        em += np.random.normal(0.0, 0.1, size=[word_embedding_dim])
                em = em / len(num_word)
                embeddings.append(em)
                vecab[k] = len(vecab)

    #将填充词加到词典最后，
    pad_id=len(vecab)
    vecab[pad_word]=pad_id
    embeddings=np.asarray(embeddings)
    embeddings=np.vstack((embeddings,np.zeros([word_embedding_dim],dtype=float)))
    return vecab,embeddings



def load_entity_desc(entity2descfilepath,entity2id):
    entityid2desc_dict={}

    entity2descfile=open(entity2descfilepath,'r')

    for line in entity2descfile:
        line=line.strip('\n').split('\t')
        # if len(line)<3:
        #     print line
        entityname=line[0]
        simple_desc=line[1].lower().split()
        #detail_desc=line[2].lower().split()
        if entity2id[entityname] not in  entityid2desc_dict:
            if len(simple_desc)>0:
                entityid2desc_dict[entity2id[entityname]] = simple_desc
            else:
                entityid2desc_dict[entity2id[entityname]] = entityname.split('_')

    for en,eid in entity2id.items():
        if  eid not in  entityid2desc_dict:
            entityid2desc_dict[eid]=en.split('_')

    entity2descfile.close()

    return entityid2desc_dict

def load_entity2id(entity2filepath):
    """
    load entity2id file
    :param entity2filepath: file path
    :return: return the dict "entity2id"
    """
    entity2id={}

    entity2file=open(entity2filepath,'r')
    for line in entity2file:
        line=line.strip('\n').split('\t')
        entity2id[line[0]]=int(line[1])
    entity2file.close()
    return entity2id

def pad_descs(entityid2desc_dict,max_entity_desc_length,pad_word):
    for k,v in entityid2desc_dict.items():
        desc_len=len(v)
        if desc_len>=max_entity_desc_length:
            entityid2desc_dict[k]=v[:max_entity_desc_length]
        else:
            padding_len=max_entity_desc_length-desc_len
            v.extend([pad_word]*padding_len)
            entityid2desc_dict[k]=v

def convert_desc_into_index(entityid2desc_dict,vecab):
    enid2desc_index={}
    for eid,desc in entityid2desc_dict.items():
        desc_id=[]
        if desc==None:
            print(eid)
        for w in desc:
            if w in vecab:
                desc_id.append(vecab[w])
            else:
                vecab[w]=len(vecab)
                desc_id.append(vecab[w])
        enid2desc_index[eid]=desc_id
    return enid2desc_index

def get_en_to_desc_id(train_en1_id,train_en2_id,enid2desc_id):
    en1_to_desc_id=[]
    en2_to_desc_id=[]

    for eid in train_en1_id:
        en1_to_desc_id.append(enid2desc_id[eid])
    for eid in train_en2_id:
        en2_to_desc_id.append(enid2desc_id[eid])
    return en1_to_desc_id,en2_to_desc_id

def get_entity_to_descid(tripletfilepath,entity2id,entityid2desc_dict, word2id,id2word,is_training=True):
    en1_to_desc_id=[]
    en2_to_desc_id = []

    tripletfile=open(tripletfilepath,'r')
    for line in tripletfile:
        line=line.strip('\n').split('\t')
        en1=line[0]
        en2=line[2]
        if  entity2id[en1] in entityid2desc_dict :
            desc=entityid2desc_dict[entity2id[en1]]
            desc_id=[]
            for w in desc:
                if is_training:
                    if w not in word2id:
                        word2id[w]=len(word2id)
                        id2word[len(id2word)]=w
                    desc_id.append(word2id[w])
                else:
                    if w  not in  word2id:
                        w='UNKNOWN_TOKEN'
                    desc_id.append(word2id[w])

            en1_to_desc_id.append(desc_id)
        else:
            raise Exception('the entity: %s not exist in entityid2desc_dict dic'.format(en1))

        if entity2id[en2] in entityid2desc_dict:
            desc=entityid2desc_dict[entity2id[en2]]
            desc_id=[]
            for w in desc:
                if is_training:
                    if w not in  word2id:
                        word2id[w]=len(word2id)
                        id2word[len(id2word)]=w
                    desc_id.append(word2id[w])
                else:
                    if w not in  word2id:
                        w='UNKNOWN_TOKEN'
                    desc_id.append(word2id[w])

            en2_to_desc_id.append(desc_id)
        else:
            raise Exception('the entity: %s not exist in enid2desc_id dic'.format(en2))

    tripletfile.close()
    en1_to_desc_id=np.array(en1_to_desc_id)
    en2_to_desc_id = np.array(en2_to_desc_id)
    return en1_to_desc_id,en2_to_desc_id



