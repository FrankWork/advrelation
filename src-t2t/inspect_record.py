# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Inspect a TFRecord file of tensorflow.Example and show tokenizations.

python data_generators/inspect.py \
    --logtostderr \
    --print_targets \
    --subword_text_encoder_filename=$DATA_DIR/vocab.endefr.8192 \
    --input_filename=$DATA_DIR/wmt_ende_tokens_8k-train-00000-of-00100
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import

import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("t2t_usr_dir", "",
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory, used if --generate_data.")

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

def length_statistics(length):
  '''get maximum, mean, quantile from length
  Args:
    length: list<int>
  '''
  length = sorted(length)
  length = np.asarray(length)

  # p7 = np.percentile(length, 70)
  # Probability{length < p7} = 0.7
  percent = [50, 70, 80, 90, 95, 98, 100]
  quantile = [np.percentile(length, p) for p in percent]
  
  print('(percent, quantile)', list(zip(percent, quantile)))

def main(_):
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  # Generate data if requested.
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)

  problem_name = FLAGS.problems
  tf.logging.info("Generating data for %s" % problem_name)
  problem = registry.problem(problem_name)
  length = problem.get_length(data_dir, tmp_dir)

  length_statistics(length)


if __name__ == "__main__":
  tf.app.run()
