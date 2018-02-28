
in_file = 'data/nyt2010/train.txt'
out_file = 'data/nyt2010/unsupervised.txt'

def data_generator():
  with open(in_file) as f:
    for i, line in enumerate(f):
      # if i >= 50:
      #   break
      segments = line.split('\t')
      e1 = segments[2]
      e2 = segments[3]
      sentence = segments[5].strip(' ###END###\n')
      yield e1, e2, sentence

def find_pos(entity, sent):
  ''' find entity position in sentence'''
  n = len(entity)
  for i in range(len(sent)):
    if sent[i:i+n]==entity:
      first, last = i, i+n-1
      return (first, last)
  return None, None

with open(out_file, 'w') as f:
  for trip in data_generator():
    e1, e2, sent_str = trip
    e1 = e1.split('_')
    e2 = e2.split('_')

    sent_toks = []
    for token in sent_str.split():
      if '_' in token:
        for tok in token.split('_'):
          sent_toks.append(tok)
      else:
        sent_toks.append(token)

    e1_first, e1_last = find_pos(e1, sent_toks)
    e2_first, e2_last = find_pos(e2, sent_toks)
    f.write('0 %d %d %d %d %s\n'%(e1_first, e1_last, e2_first, e2_last, ' '.join(sent_toks)))
