"""Keras utilities for EMTF++."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from emtf_utils import *

# tensorflow
import tensorflow as tf
try:
  _ = tf.version.VERSION
except:
  raise AssertionError('This script requires tensorflow>=2.')


# ______________________________________________________________________________
# Functions

def save_model(model, name=None, custom_objects=None):
  # Save as model.h5, model_weights.h5, and model.json
  if name is None:
    name = model.name
  model.save(name + '.h5')
  model.save_weights(name + '_weights.h5')
  with open(name + '.json', 'w') as f:
    f.write(model.to_json())
  import pickle
  if custom_objects is not None:
    with open(name + '_objects.pkl', 'wb') as f:
      pickle.dump(custom_objects, f, protocol=pickle.HIGHEST_PROTOCOL)
  return

def load_model(name, w_name, obj_name=None):
  # example usage:
  #   loaded_model = load_model('model.json', 'model_weights.h5')
  if not name.endswith('.json'):
    raise ValueError('Expected a .json file, got: {}'.format(name))
  if not w_name.endswith('.h5'):
    raise ValueError('Expected a .h5 file, got: {}'.format(w_name))
  if obj_name is not None and not obj_name.endswith('.pkl'):
    raise ValueError('Expected a .pkl file, got: {}'.format(obj_name))
  import pickle
  if obj_name is not None:
    with open(obj_name, 'rb') as f:
      custom_objects = pickle.load(f)
      tf.keras.utils.get_custom_objects().update(custom_objects)
  import json
  with open(name, 'r') as f:
    json_string = json.dumps(json.load(f))
    model = tf.keras.models.model_from_json(json_string)
  model.load_weights(w_name)
  return model
