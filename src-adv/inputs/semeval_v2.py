import re
import os
import random

from inputs import dataset


SEMEVAL_DATA_DIR = "data/SemEval"
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
LABEL_FILE = "relations.txt"

OUTPUT_DIR = "data/generated"
TRAIN_RECORD = "train.semeval.tfrecord"
TEST_RECORD = "test.semeval.tfrecord"
RESULTS_FILE = "results.txt"

VOCAB_SIZE = None#2**13 # 8k, 22k
VOCAB_FILE = "semeval.vocab"

class RawDataGenerator(object):
  '''Load text from file'''
  def __init__(self):
    self.entity_finder = re.compile(r"<e[12]>(.*?)</e[12]>")
    self.entity_tag_mask = re.compile(r"</?e[12]>")
    self.space_mask = re.compile(r'\s{2,}')

  def find_start_position(self, entities, sentence):
    ''' find start position of the entity in sentence
    Args:
      entities: a list of 2 entities, each entity is a list of tokens
      sentence: a list of tokens
    '''
    pos = []
    for entity in entities:
      n = len(entity)
      for i in range(len(sentence)):
        if sentence[i:i+n]==entity:
          # first, last = i, i+n-1
          pos.append(i)
    return pos

  def relative_distance(self, n):
    '''convert relative distance to positive number
    -60), [-60, 60], (60
    '''
    if n < -60:
      return 0
    elif n >= -60 and n <= 60:
      return n + 61
    
    return 122

  def position_feature(self, ent_pos, sentence):
    '''
    Args:
      ent_pos: int, start position of the entity
      sentence: a list of tokens
    '''
    length = len(sentence)
    return [self.relative_distance(i-ent_pos) for i in range(length)]
  
  def entity_context(self, ent_pos, sentence):
    ''' return [w(e-1), w(e), w(e+1)]
    '''
    context = []
    context.append(sentence[ent_pos])

    if ent_pos >= 1:
      context.append(sentence[ent_pos-1])
    else:
      context.append(sentence[ent_pos])
    
    if ent_pos < len(sentence)-1:
      context.append(sentence[ent_pos+1])
    else:
      context.append(sentence[ent_pos])
    
    return context 
  
  def lexical_feature(self, entities_pos, sentence):
    context1 = self.entity_context(entities_pos[0], sentence)
    context2 = self.entity_context(entities_pos[1], sentence)

    # ignore WordNet hypernyms in paper
    return context1 + context2

  def generator(self, data_files, for_vocab=False):
    '''load raw data from text file, 
    file contents:
      1	"The ... an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
      Component-Whole(e2,e1)
      Comment: Not a collection: there is structure here, organisation.

      EOF
    Returns:
      label: string
      entities: list of string, len == 2
      sentence: string
    '''
    if not isinstance(data_files, list):
      data_files = [data_files]
    
    for data_file in data_files:
      lines = open(data_file).readlines()
      n = len(lines)
      assert n % 4 == 0
      for i in range(n//4):
        text = lines[4*i].split('\t')[1].strip('"|\n')
        sentence = self.entity_tag_mask.sub(' ', text)
        sentence = self.space_mask.sub(" ", sentence)

        if for_vocab:
          yield sentence
        else:
          entities = self.entity_finder.findall(text)
          assert len(entities) == 2

          label = lines[4*i+1].strip()

          yield label, entities, sentence

class SemEvalCleanedTextData(dataset.TextDataset):

  def token_generator(self, file):
    with open(filename) as f:
      for line in f:
        words = line.strip().split(' ')
        
        for token in words[5:]:
          yield token

  def example_generator(self, file):
    with open(filename) as f:
      for line in f:
        words = line.strip().split(' ')
        
        sent = words[5:]
        sent = self.vocab_mgr.map_token_to_id(sent)

        label = int(words[0])

        entity1 = (int(words[1]), int(words[2]))
        entity2 = (int(words[3]), int(words[4]))

        example = (label, entity1, entity2, sent)
        yield example
    

class SemEvalCleanedRecordData(dataset.RecordDataset):

  def example_generator(self, raw_example_generator):
    """Generate examples."""
    raise NotImplementedError
  


_vocab_mgr = dataset.VocabMgr(data_dir, out_dir, vocab_file)
_text_data = SemEvalCleadTextData(data_dir, train_file, test_file, vocab_mgr)
_text_data.generate_vocab()

_dataset = SemEval2010Task8()

def generate_data():
  _dataset.generate_vocab()
  _dataset.generate_data()
  length = _dataset.get_length()
  length_statistics(length)

def read_data(epoch, batch_size):
  return _dataset.read_data(epoch, batch_size)

def write_results(predictions):
  label_encoder = text_encoder.ClassLabelEncoder(class_labels_fname=LABEL_FILE)
  
  start_no = 8001
  with open(FLAGS.results_file, 'w') as f:
    for idx, id in enumerate(predictions):
      if idx < 2717:
        rel = label_encoder.decode(id)
        f.write('%d\t%s\n' % (start_no+idx, rel))


