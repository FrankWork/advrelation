train_cln = 'data/SemEval/train.cln'
train_txt = 'data/SemEval/train.txt'


with open(train_cln) as f:
  for line in f:
    tokens = line.strip().split()
    e1_first = int(tokens[1])
    e1_last = int(tokens[2])
    e2_first = int(tokens[3])
    e2_last = int(tokens[4])

    e1 = tokens[5:][e1_first:e1_last+1]
    e2 = tokens[5:][e2_first:e2_last+1]

    print('%s||%s' % (' '.join(e1), ' '.join(e2)))

