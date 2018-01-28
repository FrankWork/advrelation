import os
import re

data_dir = '/home/renfeiliang/lzh/advrelation/data/SemEval/'
train_file = os.path.join(data_dir, 'train.txt')
test_file = os.path.join(data_dir, 'test.txt')

entity_finder = re.compile(r"<e[12]>(.*?)</e[12]>")

def extract(in_file, out_file):
  ents = []
  lines = open(in_file).readlines()
  n = len(lines)
  assert n % 4 == 0
  for i in range(n//4):
    text = lines[4*i].split('\t')[1].strip('"|\n')
    ent_pair = entity_finder.findall(text)
    assert len(ent_pair) == 2
    ents.append(ent_pair)
  
  with open(out_file, 'w') as f:
    for ent_pair in ents:
      e1, e2 = ent_pair[0], ent_pair[1]
      f.write(e1 + '|||' + e2 + '\n')
  
extract(train_file, 'train_ent.txt')
extract(test_file, 'test_ent.txt')