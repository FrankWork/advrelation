""" Problem definition for word to dictionary definition.
"""

import os
import re

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import generator_utils

from tensor2tensor.utils import registry

# datasets
LOCATION_OF_DATA = os.path.join(os.environ['ROOT_DIR'], "data/SemEval/")

TRAIN_DATASETS = os.path.join(LOCATION_OF_DATA, 'train.txt')
TEST_DATASETS =  os.path.join(LOCATION_OF_DATA, 'test.txt')
LABEL_FILE = os.path.join(LOCATION_OF_DATA, 'relations.txt')

class RawDataGenerator(object):
  '''Load text from file'''
  def __init__(self):
    self.entity_finder = re.compile(r"<e[12]>(.*?)</e[12]>")
    self.entity_tag_mask = re.compile(r"</?e[12]>")

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

  def generator(self, data_file, for_vocab=False):
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
    lines = open(data_file).readlines()
    n = len(lines)
    assert n % 4 == 0
    for i in range(n//4):
      text = lines[4*i].split('\t')[1].strip('"|\n')
      sentence = self.entity_tag_mask.sub(' ', text)
      
      if for_vocab:
        yield sentence
      else:
        entities = self.entity_finder.findall(text)
        assert len(entities) == 2

        label = lines[4*i+1].strip()

        yield label, entities, sentence


@registry.register_problem()
class AdvSemEval(problem.Problem):
  @property
  def num_shards(self):
    return 10

  @property
  def vocab_file(self):
    return "semeval.vocab"
  
  @property
  def targeted_vocab_size(self):
    return 2**13 # 8k, 22k

  def generator(self, data_dir, tmp_dir, train):
    """Generate examples."""
    data_file = TRAIN_DATASETS if train else TEST_DATASETS

    # Generate vocab
    raw_gen = RawDataGenerator()
    
    vocab_encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        raw_gen.generator(data_file, for_vocab=True))
    label_encoder = text_encoder.ClassLabelEncoder(class_labels_fname=LABEL_FILE)

    # Generate examples
    for label, entities, sentence in raw_gen.generator(data_file):
      entities = [vocab_encoder.encode(e) for e in entities]
      sentence = vocab_encoder.encode(sentence)

      entities_pos = raw_gen.find_start_position(entities, sentence)
      
      yield {
          "inputs": sentence,
          "targets": [label_encoder.encode(label)],
          'lexicail': raw_gen.lexical_feature(entities_pos, sentence),
          'position1': raw_gen.position_feature(entities_pos[0], sentence),
          'position2': raw_gen.position_feature(entities_pos[1], sentence),
      }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(data_dir, 1, shuffled=False)
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True), train_paths,
        self.generator(data_dir, tmp_dir, False), dev_paths)
  
  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    source_vocab_size = self._encoders["inputs"].vocab_size
    p.input_modality = {
        "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
    }
    p.target_modality = (registry.Modalities.CLASS_LABEL, 2)
    p.input_space_id = problem.SpaceID.EN_TOK
    p.target_space_id = problem.SpaceID.GENERIC
  
  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    return {
        "inputs": encoder,
        "targets": text_encoder.ClassLabelEncoder(["neg", "pos"]),
    }
  
  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.FixedLenFeature([1], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def eval_metrics(self):
    return [metrics.Metrics.ACC]
