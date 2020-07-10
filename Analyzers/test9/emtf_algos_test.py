"""Tests for functions in emtf_algos.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import pytest
except ImportError as e:
  print("ERROR: Could not import pytest. Do 'pip install --user pytest' to install it.\n")
  raise

import numpy as np

from emtf_algos import *


# ______________________________________________________________________________
def test_find_emtf_layer():
  _type = 1
  station = 1
  ring = 4
  assert find_emtf_layer(_type, station, ring) == 0

  _type = 3
  station = 1
  ring = 1
  assert find_emtf_layer(_type, station, ring) == 9

  _type = 2
  station = 2
  ring = 2
  assert find_emtf_layer(_type, station, ring) == 6

  _type = 2
  station = 1
  ring = 1
  assert find_emtf_layer(_type, station, ring) == -99

  _type = 4
  station = 4
  ring = 1
  assert find_emtf_layer(_type, station, ring) == -99

  _type = np.array([1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,4])
  station = np.array([1,1,1,1,2,2,3,3,4,4,1,1,2,2,3,3,3,4,4,4,1,2,1])
  ring = np.array([4,1,2,3,1,2,1,2,1,2,2,3,2,3,1,2,3,1,2,3,1,1,1])
  answer = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,7,8,8,8,9,10,11])
  assert (find_emtf_layer(_type, station, ring) == answer).all()

def test_find_emtf_ri_layer():
  _type = 1
  station = 1
  ring = 4
  assert find_emtf_ri_layer(_type, station, ring) == 0

  _type = 3
  station = 1
  ring = 1
  assert find_emtf_ri_layer(_type, station, ring) == 9

  _type = 2
  station = 2
  ring = 2
  assert find_emtf_ri_layer(_type, station, ring) == 13

  _type = 2
  station = 1
  ring = 1
  assert find_emtf_ri_layer(_type, station, ring) == -99

  _type = 4
  station = 4
  ring = 1
  assert find_emtf_ri_layer(_type, station, ring) == -99

  _type = np.array([1,1,1,1,1,1,1,1,1,1,3,2,2,3,2,2,2,2,2,2,2,2,4])
  station = np.array([1,1,1,1,2,2,3,3,4,4,1,1,1,2,2,2,3,3,3,4,4,4,1])
  ring = np.array([4,1,2,3,1,2,1,2,1,2,1,2,3,1,2,3,1,2,3,1,2,3,1])
  answer = np.array([0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,14,15,15,16,17,17,18])
  assert (find_emtf_ri_layer(_type, station, ring) == answer).all()

def test_decode_emtf_layer():
  assert decode_emtf_layer(0) == (1, 1, 1)
  assert decode_emtf_layer(9) == (3, 1, 1)
  assert decode_emtf_layer(6) == (2, 2, 2)

  emtf_layers = np.arange(12)
  result = decode_emtf_layer(emtf_layers)
  _type = np.array([1,1,1,1,1,2,2,2,2,3,3,4])
  station = np.array([1,1,2,3,4,1,2,3,4,1,2,1])
  ring = np.array([1,2,1,1,1,2,2,1,1,1,1,1])
  assert (result[0] == _type).all()
  assert (result[1] == station).all()
  assert (result[2] == ring).all()

def test_decode_emtf_ri_layer():
  assert decode_emtf_ri_layer(0) == (1, 1, 1)
  assert decode_emtf_ri_layer(9) == (3, 1, 1)
  assert decode_emtf_ri_layer(13) == (2, 2, 2)

  emtf_ri_layers = np.arange(19)
  result = decode_emtf_ri_layer(emtf_ri_layers)
  _type = np.array([1,1,1,1,1,1,1,1,1,3,2,2,3,2,2,2,2,2,4])
  station = np.array([1,1,1,2,2,3,3,4,4,1,1,1,2,2,3,3,4,4,1])
  ring = np.array([1,2,3,1,2,1,2,1,2,1,2,3,1,2,1,2,1,2,1])
  #assert (result[0] == _type).all()
  #assert (result[1] == station).all()
  #assert (result[2] == ring).all()
