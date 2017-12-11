import collections
import os

Raw_Example = collections.namedtuple('Raw_Example', 'label sentence')

def _load_raw_data_from_dir(dir, neg_or_pos):
  assert neg_or_pos in ('neg', 'pos')

  data = []
  dir = os.path.join(dir, neg_or_pos)
  label = True if neg_or_pos == 'pos' else False

  for filename in os.listdir(dir):
    filename = os.path.join(dir, filename)
    with open(filename) as f:
      lines = f.readlines()
      assert len(lines) == 1
      tokens = lines[0].lower().strip().split(' ')

      example = Raw_Example(label, tokens)
      data.append(example)
  return data

def _load_neg_pos_data(dir):
  pos_data = _load_raw_data_from_dir(dir, 'pos')
  neg_data = _load_raw_data_from_dir(dir, 'neg')
  data = pos_data + neg_data
  return data

def count(dir):
  length = [0]*30
  data = _load_neg_pos_data(dir)
  for example in data:
    n = len(example.sentence)
    i = n // 100
    length[i] += 1
  for i, n in enumerate(length):
    print(i, n)
    

print("train:")
count("data/aclImdb/train")
print("\n\ntest:")
count("data/aclImdb/train")


# train:
# 0 2926
# 1 11654
# 2 4665
# 3 2384
# 4 1341
# 5 786
# 6 465
# 7 306
# 8 201
# 9 212
# 10 40
# 11 5
# 12 3
# 13 4
# 14 1
# 15 2
# 16 1
# 17 1
# 18 2
# 19 0
# 20 0
# 21 0
# 22 0
# 23 0
# 24 1