""" Problem definition for word to dictionary definition.
"""

import os
import re

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import generator_utils

from tensor2tensor.utils import registry

# datasets
LOCATION_OF_DATA = "data/SemEval/"

TRAIN_DATASETS = os.path.join(LOCATION_OF_DATA, 'train.txt')
TEST_DATASETS =  os.path.join(LOCATION_OF_DATA, 'test.txt')

class RawDataGenerator(object):
  '''Load text from file'''
  def __init__(self):
    self.entity_finder = re.compile(r"<e[12]>(.*?)</e[12]>")
    self.entity_tag_mask = re.compile(r"</?e[12]>")

  def find_start_position(self, entity, sentence):
    ''' find start position of the entity in sentence
    Args:
      entity: a list of tokens
      sentence: a list of tokens
    '''
    n = len(entity)
    for i in range(len(sentence)):
      if sentence[i:i+n]==entity:
        # first, last = i, i+n-1
        return i
    return -1

  def generator(self, data_file, for_vocab=False):
    '''load raw data from text file, 
    file contents:
      1	"The ... an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
      Component-Whole(e2,e1)
      Comment: Not a collection: there is structure here, organisation.

      EOF
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
    
    text_encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        raw_gen.generator(data_file, for_vocab=True))

    # Generate examples
    for label, entities, sentence in raw_gen.generator(data_file):
      entities = [text_encoder.encode(e) for e in entities]
      sentence = text_encoder.encode(sentence)
      raw_gen.find_start_position()
      yield {
          "inputs": text_encoder.encode(doc) + [EOS],
          "targets": [int(label)],
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
