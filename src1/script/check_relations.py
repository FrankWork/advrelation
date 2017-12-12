relations = []
with open('data/SemEval/relations.txt') as f:
  for line in f:
    segment = line.strip().split()
    relations.append(segment[1])

keys = []
with open('data/SemEval/test_keys.txt') as f:
  for line in f:
    segment = line.strip().split()
    keys.append(segment[1])

data = []
with open('data/SemEval/test.cln') as f:
  for line in f:
    segment = line.strip().split()
    data.append(int(segment[0]))

assert len(keys) == len(data)
print(len(keys))

for i in range(len(keys)):
  id = data[i]
  assert relations[id] == keys[i]
print('done!')