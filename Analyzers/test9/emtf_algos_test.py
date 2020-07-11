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
  _type, station, ring = 1, 1, 4
  assert find_emtf_layer(_type, station, ring) == 0

  _type, station, ring = 3, 1, 1
  assert find_emtf_layer(_type, station, ring) == 9

  _type, station, ring = 2, 2, 2
  assert find_emtf_layer(_type, station, ring) == 6

  _type, station, ring = 2, 1, 1
  assert find_emtf_layer(_type, station, ring) == -99

  _type, station, ring = 4, 4, 1
  assert find_emtf_layer(_type, station, ring) == -99

  with pytest.raises(ValueError):
    _type, station, ring = 5, 5, 5
    find_emtf_layer(_type, station, ring)

  _type = np.array([1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,4])
  station = np.array([1,1,1,1,2,2,3,3,4,4,1,1,2,2,3,3,3,4,4,4,1,2,1])
  ring = np.array([4,1,2,3,1,2,1,2,1,2,2,3,2,3,1,2,3,1,2,3,1,1,1])
  answer = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,7,8,8,8,9,10,11])
  assert (find_emtf_layer(_type, station, ring) == answer).all()

def test_find_emtf_ri_layer():
  _type, station, ring = 1, 1, 4
  assert find_emtf_ri_layer(_type, station, ring) == 0

  _type, station, ring = 3, 1, 1
  assert find_emtf_ri_layer(_type, station, ring) == 9

  _type, station, ring = 2, 2, 2
  assert find_emtf_ri_layer(_type, station, ring) == 13

  _type, station, ring = 2, 1, 1
  assert find_emtf_ri_layer(_type, station, ring) == -99

  _type, station, ring = 4, 4, 1
  assert find_emtf_ri_layer(_type, station, ring) == -99

  with pytest.raises(ValueError):
    _type, station, ring = 5, 5, 5
    find_emtf_layer(_type, station, ring)

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
  assert (result[0] == _type).all()
  assert (result[1] == station).all()
  assert (result[2] == ring).all()

def test_find_emtf_chamber():
  _type, station, cscid, subsector, neighbor = 1, 1, 3, 1, 0
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == 2

  _type, station, cscid, subsector, neighbor = 3, 1, 3, 1, 0
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == 2 + 54

  _type, station, cscid, subsector, neighbor = 1, 4, 9, 0, 0
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == 9 + 9 + 9 + 9 + 8

  _type, station, cscid, subsector, neighbor = 2, 4, 9, 0, 0
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == 9 + 9 + 9 + 9 + 8 + 54

  _type, station, cscid, subsector, neighbor = 1, 1, 3, 1, 1
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == 45

  _type, station, cscid, subsector, neighbor = 3, 1, 3, 1, 1
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == 45 + 54

  _type, station, cscid, subsector, neighbor = 4, 1, 3, 0, 1
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == 3 + 54 + 54

  _type, station, cscid, subsector, neighbor = 1, 1, 3, 0, 0
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == -99

  with pytest.raises(ValueError):
    _type, station, cscid, subsector, neighbor = 5, 5, 3, 1, 0
    find_emtf_chamber(_type, station, cscid, subsector, neighbor)

  _type = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
  station = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
  cscid = np.array([1,2,3,4,5,6,7,8,9,3,6,9])
  subsector = np.array([2,2,2,2,2,2,2,2,2,2,2,2])
  neighbor = np.array([0,0,0,0,0,0,0,0,0,1,1,1])
  answer = np.array([9,10,11,12,13,14,15,16,17,45,46,47])
  assert (find_emtf_chamber(_type, station, cscid, subsector, neighbor) == answer).all()

def test_decode_emtf_chamber():
  pass

def test_find_emtf_zones():
  def packbits(a, b, c):
    return (a*4) + (b*2) + (c*1)

  ri_layer, emtf_theta = 0, 10
  assert find_emtf_zones(ri_layer, emtf_theta) == packbits(1, 0, 0)

  ri_layer, emtf_theta = 18, 10
  assert find_emtf_zones(ri_layer, emtf_theta) == packbits(1, 0, 0)

  with pytest.raises(IndexError):
    ri_layer, emtf_theta = 99, 10
    find_emtf_zones(ri_layer, emtf_theta)

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  emtf_theta = np.array([10]*9)
  answer = np.array([4,0,0,4,0,4,0,4,0])
  assert (find_emtf_zones(ri_layer, emtf_theta) == answer).all()

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  emtf_theta = np.array([25]*9)
  answer = np.array([6,0,0,6,0,6,0,6,0])
  assert (find_emtf_zones(ri_layer, emtf_theta) == answer).all()

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  emtf_theta = np.array([46]*9)
  answer = np.array([2,2,0,2,0,0,2,0,2])
  assert (find_emtf_zones(ri_layer, emtf_theta) == answer).all()

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  emtf_theta = np.array([53]*9)
  answer = np.array([2,3,0,0,1,0,3,0,3])
  assert (find_emtf_zones(ri_layer, emtf_theta) == answer).all()

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  emtf_theta = np.array([84]*9)
  answer = np.array([0,1,0,0,1,0,1,0,1])
  assert (find_emtf_zones(ri_layer, emtf_theta) == answer).all()

def test_find_emtf_timezones():
  def packbits(a, b, c):
    return (a*4) + (b*2) + (c*1)

  ri_layer, bx = 0, -1
  assert find_emtf_timezones(ri_layer, bx) == packbits(1, 1, 0)

  ri_layer, bx = 0, 0
  assert find_emtf_timezones(ri_layer, bx) == packbits(0, 1, 1)

  ri_layer, bx = 0, 1
  assert find_emtf_timezones(ri_layer, bx) == packbits(0, 0, 1)

  with pytest.raises(IndexError):
    ri_layer, bx = 99, -1
    find_emtf_zones(ri_layer, bx)

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  bx = np.array([-1]*9)
  answer = np.array([6]*9)
  assert (find_emtf_timezones(ri_layer, bx) == answer).all()

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  bx = np.array([0]*9)
  answer = np.array([3]*9)
  assert (find_emtf_timezones(ri_layer, bx) == answer).all()

  ri_layer = np.array([9,10,11,12,13,14,15,16,17])
  bx = np.array([0]*9)
  answer = np.array([2]*9)
  assert (find_emtf_timezones(ri_layer, bx) == answer).all()

def test_find_emtf_zo_layer():
  ri_layer, zone = 0, 0
  assert find_emtf_zo_layer(ri_layer, zone) == 2

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  zone = 0
  answer = np.array([2,-99,-99,4,-99,5,-99,7,-99])
  assert (find_emtf_zo_layer(ri_layer, zone) == answer).all()

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  zone = 1
  answer = np.array([1,2,-99,4,-99,5,5,7,7])
  assert (find_emtf_zo_layer(ri_layer, zone) == answer).all()

  ri_layer = np.array([0,1,2,3,4,5,6,7,8])
  zone = 2
  answer = np.array([-99,0,-99,-99,3,-99,4,-99,6])
  assert (find_emtf_zo_layer(ri_layer, zone) == answer).all()
