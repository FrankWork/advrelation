import os

data_dir = '/home/frank/work/KnowledgeGraph/advrelation/data'
train_file = os.path.join(data_dir, 'train_nopos_ty=6.txt')
test_file = os.path.join(data_dir, 'test_nopos_ty=6.txt')

def extract(in_file, out_file):
  ents = []
  with open(in_file) as f:
    for line in f:
      tokens = line.lower().strip().split()

      e1_first, e1_last  = (int(tokens[1]), int(tokens[2]))
      e2_first, e2_last  = (int(tokens[3]), int(tokens[4]))

      sentence = tokens[5:]
      e1 = sentence[e1_first:e1_last+1]
      e2 = sentence[e2_first:e2_last+1]

      ents.append((e1, e2))
  
  with open(out_file, 'w') as f:
    for ent in ents:
      e1, e2 = ent
      f.write(' '.join(e1) + '|||' + ' '.join(e2) + '\n')
  
extract(train_file, 'train_ent.txt')
extract(test_file, 'test_ent.txt')