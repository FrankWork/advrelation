from inputs import dataset

DATA_DIR = 'data/nyt2010'
UNSUP_FILE = 'unsupervised.txt'

class NYT2010CleanedTextData(dataset.TextDataset):

  def token_generator(self, file):
    with open(file) as f:
      for line in f:
        words = line.strip().split(' ')
        
        for token in words[5:]:
          yield token

  def get_length(self):
    length = []
    with open(self.unsup_file) as f:
      for line in f:
        words = line.strip().split(' ')
        n = len(words[5:])
        length.append(n)
    return length

  def example_generator(self, file):
    with open(file) as f:
      for line in f:
        words = line.strip().split(' ')
        
        sent = words[5:]
        sent = self.vocab_mgr.map_token_to_id(sent)

        label = int(words[0])

        entity1 = (int(words[1]), int(words[2]))
        entity2 = (int(words[3]), int(words[4]))

        example = (label, entity1, entity2, sent)
        yield example

_nyt_text = NYT2010CleanedTextData(DATA_DIR, None, unsup_file=UNSUP_FILE)

def length_statistics():
  _nyt_text.length_statistics()
  