# The following source code is obtained from:
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_configs.py
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py
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
"""Interface for a layer to express how to quantize it."""

from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras import quantizers


class DefaultDenseQuantizeConfig(quantize_config.QuantizeConfig):
  """QuantizeConfig which quantizes the weights and activations of a layer."""

  def get_weights_and_quantizers(self, layer):
    weight_quantizer = quantizers.LastValueQuantizer(
        num_bits=8, per_axis=False, symmetric=True, narrow_range=True)
    return [(layer.kernel, weight_quantizer)]

  def get_activations_and_quantizers(self, layer):
    activation_quantizer = quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)
    return [(layer.activation, activation_quantizer)]

  def set_quantize_weights(self, layer, quantize_weights):
    layer.kernel = quantize_weights[0]

  def set_quantize_activations(self, layer, quantize_activations):
    layer.activation = quantize_activations[0]

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}


class DefaultOutputQuantizeConfig(quantize_config.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    output_quantizer = quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)
    return [output_quantizer]

  def get_config(self):
    return {}


class NoOpQuantizeConfig(quantize_config.QuantizeConfig):
  """QuantizeConfig which does not quantize any part of the layer."""

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}
