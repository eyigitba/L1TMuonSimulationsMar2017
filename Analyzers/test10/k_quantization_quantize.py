# The following source code is obtained from:
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize.py#L80-L210
# ==============================================================================

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Quantization API functions for tf.keras models."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quantize_annotate_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize as quantize_mod

keras = tf.keras


def _add_quant_wrapper(layer):
  """Add annotation wrapper."""
  # Already annotated layer. No need to wrap.
  if isinstance(layer, quantize_annotate_mod.QuantizeAnnotate):
    return layer
  if isinstance(layer, tf.keras.Model):
    raise ValueError(
        'Quantizing a tf.keras Model inside another tf.keras Model is not supported.')
  return quantize_annotate_mod.QuantizeAnnotate(layer)


def quantize_model(to_quantize, quantize_annotate_fn=_add_quant_wrapper):
  """Quantize a `tf.keras` model with the default quantization implementation.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

  For more information, see
  https://www.tensorflow.org/model_optimization/guide/quantization/training

  Quantize a model:

  ```python
  # Quantize sequential model
  model = quantize_model(
      keras.Sequential([
          layers.Dense(10, activation='relu', input_shape=(100,)),
          layers.Dense(2, activation='sigmoid')
      ]))

  # Quantize functional model
  in = tf.keras.Input((3,))
  out = tf.keras.Dense(2)(in)
  model = tf.keras.Model(in, out)

  quantized_model = quantize_model(model)
  ```

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    to_quantize: tf.keras model to be quantized. It can have pre-trained
      weights.

  Returns:
    Returns a new `tf.keras` model prepared for quantization.
  """
  if to_quantize is None:
    raise ValueError('`to_quantize` cannot be None')

  if not isinstance(to_quantize, keras.Model):
    raise ValueError(
        '`to_quantize` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers.'
        'You passed an instance of type: {input}.'.format(
            input=to_quantize.__class__.__name__))

  if not isinstance(to_quantize, keras.Sequential) \
      and not to_quantize._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_quantize` can only either be a tf.keras Sequential or '
        'Functional model.')

  annotated_model = quantize_annotate_model(to_quantize, quantize_annotate_fn)
  return quantize_mod.quantize_apply(annotated_model)


def quantize_annotate_model(to_annotate, quantize_annotate_fn=_add_quant_wrapper):
  """Annotate a `tf.keras` model to be quantized.

  This function does not actually quantize the model. It merely specifies
  that the model needs to be quantized. `quantize_apply` can then be used
  to quantize the model.

  This function is intended to be used in conjunction with the
  `quantize_annotate_layer` API. Otherwise, it is simpler to use
  `quantize_model`.

  Annotate a model while overriding the default behavior for a layer:

  ```python
  quantize_config = MyDenseQuantizeConfig()

  model = quantize_annotate_model(
    keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(
          layers.Dense(2, activation='sigmoid'),
          quantize_config=quantize_config)
    ]))

  # The first Dense layer gets quantized with the default behavior,
  # but the second layer uses `MyDenseQuantizeConfig` for quantization.
  quantized_model = quantize_apply(model)
  ```

  Note that this function removes the optimizer from the original model.

  Args:
    to_annotate: `tf.keras` model which needs to be quantized.

  Returns:
    New tf.keras model with each layer in the model wrapped with
    `QuantizeAnnotate`. The new model preserves weights from the original
    model.
  """
  if to_annotate is None:
    raise ValueError('`to_annotate` cannot be None')

  if not isinstance(to_annotate, keras.Model):
    raise ValueError(
        '`to_annotate` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers. '
        'You passed an instance of type: {input}.'.format(
            input=to_annotate.__class__.__name__))

  if not isinstance(to_annotate, keras.Sequential) \
      and not to_annotate._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_annotate` can only either be a tf.keras Sequential or '
        'Functional model.')

  return keras.models.clone_model(
      to_annotate, input_tensors=None, clone_function=quantize_annotate_fn)
