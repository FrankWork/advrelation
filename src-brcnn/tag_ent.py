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
  pos = []
  start = 0
  e1, e2 = entities

  n = len(e1)
  e1_r = list(reversed(e1))
  for i in range(start, len(sentence)):
    sliced = sentence[i:i+n]
    if sliced==e1 or sliced==e1_r:
      first, last = i, i+n
      pos.append(first)
      pos.append(last)
      start = last
      break
  
  if len(pos) != 2:
    for token in e1:
      if len(pos) == 2:
        break
      for i in range(start, len(sentence)):
        if token == sentence[i]:
          first, last = i, i+1
          pos.append(first)
          pos.append(last)
          start = last
          break
  
  n = len(e2)
  e2_r = list(reversed(e2))
  for i in range(start, len(sentence)):
    sliced = sentence[i:i+n]
    if sliced==e2 or sliced==e2_r:
      first, last = i, i+n
      pos.append(first)
      pos.append(last)
      start = last
      break
  
  if len(pos) != 4:
    for token in e2:
      if len(pos) == 4:
        break
      for i in range(start, len(sentence)):
        if token == sentence[i]:
          first, last = i, i+1
          pos.append(first)
          pos.append(last)
          start = last
          break
  
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

  
