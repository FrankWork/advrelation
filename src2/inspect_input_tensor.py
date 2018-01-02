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

def main(_):
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  # Generate data if requested.
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)

  problem_name = FLAGS.problems
  tf.logging.info("Generating data for %s" % problem_name)
  problem = registry.problem(problem_name)
  
  hparams = registry.hparams(FLAGS.hparams_set)()
  hparams.add_hparam("data_dir", data_dir)

  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams)

  features, targets = train_input_fn(None, None)
  inputs_tensor = features['inputs']
  lexical_tensor = features['lexical']

  with tf.Session() as sess:
    for i in range(10):
      inputs, lexical = sess.run([inputs_tensor, lexical_tensor])
      print(inputs.shape, lexical.shape)



if __name__ == "__main__":
  tf.app.run()
