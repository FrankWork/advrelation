import pickle
import os

train_file = 'data/vdir/directed_train_paths'
test_file = 'data/vdir/directed_test_paths'

def find_ent_position(entities, sentence):
  ''' find start position of the entity in sentence
  Args:
    entities: a list of 2 entities, each entity is a list of tokens
    sentence: a list of tokens
  '''
  sentence = [token.lower() for token in sentence]
  pos = []
  for entity in entities:
    n = len(entity)
    entity_r = list(reversed(entity))
    
    for i in range(len(sentence)):
      sliced = sentence[i:i+n]
      if sliced==entity or sliced==entity_r:
        first, last = i, i+n
        pos.append(first)
        pos.append(last)
  return pos

def tag(pkl_file, ent_file, out_file):
  with open(pkl_file, 'rb') as f:
    word_p, _, _ = pickle.load(f)
  
  ents = []
  with open(ent_file) as f:
    for line in f:
      e1, e2 = line.strip().split('|||')
      e1 = e1.split()
      e2 = e2.split()
      ents.append((e1, e2))
  
  with open(out_file, 'w') as f:
    for entities, tokens in zip(ents, word_p):
      position = find_ent_position(entities, tokens)
      # assert len(position) == 4
      if len(position) != 4:
        print(entities)
        print(tokens)
        print(position)
        exit()

      position = [str(x) for x in position]
      f.write(' '.join(position) + '\n')

tag(train_file, 'train_ent.txt', 'train_ent_pos.txt')
tag(test_file, 'test_ent.txt', 'test_ent_pos.txt')

  
