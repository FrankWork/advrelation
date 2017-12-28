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

TRAIN_DATASETS = [ os.path.join(LOCATION_OF_DATA, 'train.txt')]
TEST_DATASETS = [ os.path.join(LOCATION_OF_DATA, 'test.txt')]

class RawDataGenerator(object):
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
    for i in range(len(tokens)):
      if tokens[i:i+n]==entity:
        # first, last = i, i+n-1
        return i
    return -1

  def _load_raw_data(filename, relation2id):
    '''load raw data from text file, 
    file contents:
      1	"The ... an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
      Component-Whole(e2,e1)
      Comment: Not a collection: there is structure here, organisation.

      2	"The <e1>child</e1> ... the <e2>cradle</e2> by means of a cord."
      Other
      Comment:

      EOF

    return: a list of Raw_Example
    '''
    data = []
    lines = open(filename).readlines()
    n = len(lines)
    assert n % 4 == 0
    for i in range(n//4):
      sentence = lines[4*i].split('\t')[1].strip('"|\n').lower()
      
      entities = _entity_regex.findall(sentence)
      assert len(entities) == 2

      sentence = _etag_mask.sub(' ', sentence)
      tokens = util.wordpunct_tokenizer(sentence)

      entities = [util.wordpunct_tokenizer(entity) for entity in entities]
      entity1 = _find_entity_pos(entities[0], tokens)
      entity2 = _find_entity_pos(entities[1], tokens)
      assert entity1 is not None and entity2 is not None

      rel_text = lines[4*i+1].strip()
      label = relation2id[rel_text]

      example = Raw_Example(label, entity1, entity2, tokens)
      data.append(example)

    return data


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
    raise 2**13 # 8k, 22k

  def doc_generator(self, imdb_dir, dataset, include_label=False):
    dirs = [(os.path.join(imdb_dir, dataset, "pos"), True), (os.path.join(
        imdb_dir, dataset, "neg"), False)]

    for d, label in dirs:
      for filename in os.listdir(d):
        with tf.gfile.Open(os.path.join(d, filename)) as imdb_f:
          doc = imdb_f.read().strip()
          if include_label:
            yield doc, label
          else:
            yield doc
 


  def generator(self, data_dir, tmp_dir, train):
    """Generate examples."""
    # Generate vocab
    encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        self.doc_generator(imdb_dir, "train"))

    # Generate examples
    dataset = "train" if train else "test"
    for doc, label in self.doc_generator(imdb_dir, dataset, include_label=True):
      yield {
          "inputs": encoder.encode(doc) + [EOS],
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
