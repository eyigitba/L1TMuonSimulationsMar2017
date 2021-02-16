#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
try:
  _ = tf.version.VERSION
except:
  raise AssertionError('This script requires tensorflow>=2.')

from tensorflow.python.framework import convert_to_constants
from tensorflow.python.module import module

# This is copied from:
# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/tests/convert_to_constants_test.py#L43-L59
def _freezeModel(model):
  """Freezes the model.

  Args:
    model: Function.

  Returns:
    root: AutoTrackable object with original ConcreteFunction.
    output_func: frozen ConcreteFunction.
  """
  root = module.Module()
  root.f = model
  input_func = root.f.get_concrete_function()

  output_func = convert_to_constants.convert_variables_to_constants_v2(
      input_func, lower_control_flow=False)
  return root, output_func

def usage():
  print('usage: python {0} FILE'.format(sys.argv[0]))
  print('')
  print('arguments:')
  print('  FILE    a model JSON file, e.g. \'model.json\'')


# ______________________________________________________________________________
if __name__ == "__main__":
  if len(sys.argv) < 2:
    usage()
    sys.exit(1)

  model_file = sys.argv[1]
  model_weights_file = model_file.replace('model', 'model_weights').replace('.json', '.h5')
  constant_graph_file = model_file.replace('model', 'model_graph').replace('.json', '.pb')

  # Load model
  from nn_models import load_my_model, update_keras_custom_objects
  update_keras_custom_objects()
  loaded_model = load_my_model(name=model_file, weights_name=model_weights_file)

  # Save as a constant graph
  @tf.function(input_signature=[
      tf.TensorSpec(shape=loaded_model.inputs[0].shape, dtype=loaded_model.inputs[0].dtype)
  ])
  def to_save(x):
    return loaded_model(x)

  _, output_func = _freezeModel(to_save)
  tf.io.write_graph(output_func.graph, './', constant_graph_file, as_text=False)
  print('constant graph: {}'.format(constant_graph_file))
  #tf.io.write_graph(output_func.graph, './', constant_graph_file + '.txt', as_text=True)
  #print('(as text): {}'.format(constant_graph_file + '.txt'))

  # Print the node names
  input_node_names = [node.op.name for node in output_func.inputs]
  output_node_names = [node.op.name for node in output_func.outputs]
  print('input_node_names: {}'.format(input_node_names))
  print('output_node_names: {}'.format(output_node_names))
