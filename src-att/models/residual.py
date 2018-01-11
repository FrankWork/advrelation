# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batch_norm_relu(inputs, is_training, data_format, decay=0.997, epsilon=1e-5):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  # inputs = tf.layers.batch_normalization(
  #     inputs=inputs, axis=1 if data_format == 'channels_first' else 2,
  #     momentum=0.997, epsilon=epsilon, center=True,
  #     scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, length] or
      [batch, length, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv1d or max_pool1d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv1d_fixed_padding(inputs, filters, kernel_size, strides, data_format='channels_last'):
  """Strided 1-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv1d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv1d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=True,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format='channels_last'):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, length] or
      [batch, length, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv1d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv1d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, length] or
      [batch, length, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv1d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv1d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv1d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, length] or
      [batch, length, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv1d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  projection_shortcut = None
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

  return tf.identity(inputs, name)


def residual_net(inputs, length, filters, is_training, resnet_size=8, 
                                                data_format='channels_last'):
  """ResNet v2 models.

  Args:
    resnet_size: A single integer for the size of the ResNet model.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.

  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  if resnet_size % 6 != 2:
    raise ValueError('resnet_size must be 6n + 2:', resnet_size)

  # num_blocks = (resnet_size - 2) // 6
  num_blocks = 1

  # model
  inputs = conv1d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = tf.identity(inputs, 'initial_conv')

  inputs = block_layer(
      inputs=inputs, filters=filters, block_fn=building_block, blocks=num_blocks,
      strides=1, is_training=is_training, name='block_layer1',
      data_format=data_format)
  # inputs = block_layer(
  #     inputs=inputs, filters=filters, block_fn=building_block, blocks=num_blocks,
  #     strides=1, is_training=is_training, name='block_layer2',
  #     data_format=data_format)
  # inputs = block_layer(
  #     inputs=inputs, filters=filters, block_fn=building_block, blocks=num_blocks,
  #     strides=2, is_training=is_training, name='block_layer3',
  #     data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = tf.layers.max_pooling1d(
      inputs=inputs, pool_size=length, strides=length, padding='SAME',
      data_format=data_format)
  inputs = tf.identity(inputs, 'final_max_pool')
  # print(inputs.shape)
  # exit()
  inputs = tf.squeeze(inputs, axis=1)
  # inputs = tf.reshape(inputs, [-1, 64])

  # inputs = tf.layers.dense(inputs=inputs, units=num_classes)
  # inputs = tf.identity(inputs, 'final_dense')

  return inputs
