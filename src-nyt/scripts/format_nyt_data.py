
in_file = 'data/nyt2010/test.txt'
out_file = 'data/nyt2010/test.cln'
relation_file = 'data/nyt2010/relation2id.txt'

def data_generator(in_file):
  with open(in_file) as f:
    for i, line in enumerate(f):
      # if i >= 50:
      #   break
      segments = line.split('\t')
      e1 = segments[2]
      e2 = segments[3]
      rel = segments[4]
      sentence = segments[5].strip(' ###END###\n')
      yield e1, e2, rel, sentence

def find_pos(entity, sent):
  ''' find entity position in sentence'''
  n = len(entity)
  for i in range(len(sent)):
    if sent[i:i+n]==entity:
      first, last = i, i+n-1
      return (first, last)
  return None, None

def format_data(in_file, out_file, rel2id):
  with open(out_file, 'w') as f:
    for item in data_generator(in_file):
      e1, e2, rel_str, sent_str = item
      e1 = e1.split('_')
      e2 = e2.split('_')

      rel_id = rel2id['NA']
      if rel_str in rel2id:
        rel_id = rel2id[rel_str]

      sent_toks = []
      for token in sent_str.split():
        if '_' in token:
          for tok in token.split('_'):
            sent_toks.append(tok)
        else:
          sent_toks.append(token)

      e1_first, e1_last = find_pos(e1, sent_toks)
      e2_first, e2_last = find_pos(e2, sent_toks)
      f.write('%d %d %d %d %d %s\n'%(rel_id, e1_first, e1_last, 
                                    e2_first, e2_last, ' '.join(sent_toks)))

rel2id = {}
with open(relation_file) as f:
  for line in f:
    segs = line.strip().split()
    rel2id[segs[0]] = int(segs[1])

format_data(in_file, out_file, rel2id)
