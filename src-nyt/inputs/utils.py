
def relative_distance(n):
  '''convert relative distance to positive number
  -60), [-60, 60], (60
  '''
  if n < -60:
    return 0
  elif n >= -60 and n <= 60:
    return n + 61
  
  return 122

def position_feature(e_first, e_last, length):
  pos = []
  if e_first >= length:
    e_first = length-1
  if e_last >= length:
    e_last = length-1

  for i in range(length):
    if i<e_first:
      pos.append(relative_distance(i-e_first))
    elif i>=e_first and i<=e_last:
      pos.append(relative_distance(0))
    else:
      pos.append(relative_distance(i-e_last))
  return pos

def write_results(predictions, label_file, relation_file):
  id2relation = []
  with open(label_file) as f:
    for id, line in enumerate(f):
      rel = line.strip()
      id2relation.append(rel)
  
  start_no = 8001
  with open(relation_file), 'w') as f:
    for idx, id in enumerate(predictions):
      if idx < 2717:
        rel = id2relation[id]
        f.write('%d\t%s\n' % (start_no+idx, rel))