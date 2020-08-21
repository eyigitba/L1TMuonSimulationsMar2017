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
  assert find_emtf_chamber(_type, station, cscid, subsector, neighbor) == 6 + 54 + 54

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

def test_chamber_to_ri_layer_lut():
  # Given y = F(x), find x = F^{-1}(y). F is assumed to be a LUT.
  inverse_fn = lambda F, y: [[i for (i, y_i) in enumerate(F) if y_i == y_j] for y_j in y]
  to_array = lambda x: np.asarray([np.asarray(x_i) for x_i in x])
  to_list = lambda x: [x_i.tolist() for x_i in x]
  flatten = lambda x: np.asarray([x_i_i for x_i in x for x_i_i in x_i])

  num_emtf_ri_layers = 19
  ri_layer_to_chamber_lut = to_array(inverse_fn(chamber_to_ri_layer_lut, range(num_emtf_ri_layers)))
  ri_layer_to_chamber_lut_flat = flatten(ri_layer_to_chamber_lut)

  assert len(chamber_to_ri_layer_lut) == num_emtf_chambers
  assert len(ri_layer_to_chamber_lut) == num_emtf_ri_layers
  assert len(ri_layer_to_chamber_lut_flat) == num_emtf_chambers
  assert sorted(ri_layer_to_chamber_lut_flat.tolist()) == list(range(num_emtf_chambers))

  # ME1/1, ME1/2, ME1/3, ME2/1, ME2/2, ME3/1, ME3/2, ME4/1, ME4/2,
  # GE1/1, RE1/2, RE1/3, GE2/1, RE2/2, RE3/1, RE3/2, RE4/1, RE4/2,
  # ME0
  ri_layer_to_chamber_lut_answer = [
    [  0,  1,  2,  9, 10, 11, 45,],
    [  3,  4,  5, 12, 13, 14, 46,],
    [  6,  7,  8, 15, 16, 17, 47,],
    [ 18, 19, 20, 48,],
    [ 21, 22, 23, 24, 25, 26, 49,],
    [ 27, 28, 29, 50,],
    [ 30, 31, 32, 33, 34, 35, 51,],
    [ 36, 37, 38, 52,],
    [ 39, 40, 41, 42, 43, 44, 53,],
    [ 54, 55, 56, 63, 64, 65, 99,],
    [ 57, 58, 59, 66, 67, 68,100,],
    [ 60, 61, 62, 69, 70, 71,101,],
    [ 72, 73, 74,102,],
    [ 75, 76, 77, 78, 79, 80,103,],
    [ 81, 82, 83,104,],
    [ 84, 85, 86, 87, 88, 89,105,],
    [ 90, 91, 92,106,],
    [ 93, 94, 95, 96, 97, 98,107,],
    [108,109,110,111,112,113,114,],
  ]
  assert to_list(ri_layer_to_chamber_lut) == ri_layer_to_chamber_lut_answer

  # Further tests (zone 0)
  zone = 0
  num_emtf_zo_layers = 8
  ri_layer_to_zo_layer_lut = find_emtf_zo_layer_lut()[:, zone]
  zo_layer_to_ri_layer_lut = to_array(inverse_fn(ri_layer_to_zo_layer_lut, range(num_emtf_zo_layers)))

  # ME0, GE1/1, ME1/1, GE2/1, ME2/1, ME3/1, RE3/1, ME4/1
  zo_layer_to_ri_layer_lut_answer = [[18], [9], [0], [12], [3], [5], [14], [7, 16]]
  assert to_list(zo_layer_to_ri_layer_lut) == zo_layer_to_ri_layer_lut_answer

  ri_layer_to_zo_bounds_lut_0 = find_emtf_zones_lut()[:, zone, 0]
  ri_layer_to_zo_bounds_lut_1 = find_emtf_zones_lut()[:, zone, 1]
  zo_layer_to_chamber_lut = to_array([
      [c for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])
  zo_layer_to_zo_bounds_lut_0 = to_array([
      [ri_layer_to_zo_bounds_lut_0[ri_layer] for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])
  zo_layer_to_zo_bounds_lut_1 = to_array([
      [ri_layer_to_zo_bounds_lut_1[ri_layer] for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])

  zo_layer_to_chamber_lut_answer = [
    [108, 109, 110, 111, 112, 113, 114],
    [ 54,  55,  56,  63,  64,  65,  99],
    [  0,   1,   2,   9,  10,  11,  45],
    [ 72,  73,  74, 102],
    [ 18,  19,  20,  48],
    [ 27,  28,  29,  50],
    [ 81,  82,  83, 104],
    [ 36,  37,  38,  52,  90,  91,  92, 106],
  ]
  assert to_list(zo_layer_to_chamber_lut) == zo_layer_to_chamber_lut_answer

  zo_layer_to_zo_bounds_lut_0_answer = [
    [ 4,  4,  4,  4,  4,  4,  4],
    [17, 17, 17, 17, 17, 17, 17],
    [ 4,  4,  4,  4,  4,  4,  4],
    [ 4,  4,  4,  4],
    [ 4,  4,  4,  4],
    [ 4,  4,  4,  4],
    [ 4,  4,  4,  4],
    [ 4,  4,  4,  4,  4,  4,  4,  4],
  ]
  assert to_list(zo_layer_to_zo_bounds_lut_0) == zo_layer_to_zo_bounds_lut_0_answer

  zo_layer_to_zo_bounds_lut_1_answer = [
    [23, 23, 23, 23, 23, 23, 23],
    [26, 26, 26, 26, 26, 26, 26],
    [26, 26, 26, 26, 26, 26, 26],
    [25, 25, 25, 25],
    [25, 25, 25, 25],
    [25, 25, 25, 25],
    [25, 25, 25, 25],
    [25, 25, 25, 25, 25, 25, 25, 25],
  ]
  assert to_list(zo_layer_to_zo_bounds_lut_1) == zo_layer_to_zo_bounds_lut_1_answer

  # Further tests (zone 1)
  zone = 1
  num_emtf_zo_layers = 8
  ri_layer_to_zo_layer_lut = find_emtf_zo_layer_lut()[:, zone]
  zo_layer_to_ri_layer_lut = to_array(inverse_fn(ri_layer_to_zo_layer_lut, range(num_emtf_zo_layers)))

  # GE1/1, ME1/1, ME1/2, GE2/1, ME2/1, ME3/1, RE3/1, ME4/1
  zo_layer_to_ri_layer_lut_answer = [[9], [0], [1, 10], [12], [3], [5, 6], [14, 15], [7, 8, 16, 17]]
  assert to_list(zo_layer_to_ri_layer_lut) == zo_layer_to_ri_layer_lut_answer

  ri_layer_to_zo_bounds_lut_0 = find_emtf_zones_lut()[:, zone, 0]
  ri_layer_to_zo_bounds_lut_1 = find_emtf_zones_lut()[:, zone, 1]
  zo_layer_to_chamber_lut = to_array([
      [c for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])
  zo_layer_to_zo_bounds_lut_0 = to_array([
      [ri_layer_to_zo_bounds_lut_0[ri_layer] for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])
  zo_layer_to_zo_bounds_lut_1 = to_array([
      [ri_layer_to_zo_bounds_lut_1[ri_layer] for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])

  zo_layer_to_chamber_lut_answer = [
    [ 54,  55,  56,  63,  64,  65,  99],
    [  0,   1,   2,   9,  10,  11,  45],
    [  3,   4,   5,  12,  13,  14,  46,  57,  58,  59,  66,  67,  68, 100],
    [ 72,  73,  74, 102],
    [ 18,  19,  20,  48],
    [ 27,  28,  29,  50,  30,  31,  32,  33,  34,  35,  51],
    [ 81,  82,  83, 104,  84,  85,  86,  87,  88,  89, 105],
    [ 36,  37,  38,  52,  39,  40,  41,  42,  43,  44,  53,  90,  91, 92, 106,  93,  94,  95,  96,  97,  98, 107],
  ]
  assert to_list(zo_layer_to_chamber_lut) == zo_layer_to_chamber_lut_answer

  zo_layer_to_zo_bounds_lut_0_answer = [
    [24, 24, 24, 24, 24, 24, 24],
    [24, 24, 24, 24, 24, 24, 24],
    [46, 46, 46, 46, 46, 46, 46, 52, 52, 52, 52, 52, 52, 52],
    [23, 23, 23, 23],
    [23, 23, 23, 23],
    [23, 23, 23, 23, 44, 44, 44, 44, 44, 44, 44],
    [23, 23, 23, 23, 40, 40, 40, 40, 40, 40, 40],
    [23, 23, 23, 23, 38, 38, 38, 38, 38, 38, 38, 23, 23, 23, 23, 36, 36, 36, 36, 36, 36, 36],
  ]
  assert to_list(zo_layer_to_zo_bounds_lut_0) == zo_layer_to_zo_bounds_lut_0_answer

  zo_layer_to_zo_bounds_lut_1_answer = [
    [52, 52, 52, 52, 52, 52, 52],
    [53, 53, 53, 53, 53, 53, 53],
    [54, 54, 54, 54, 54, 54, 54, 56, 56, 56, 56, 56, 56, 56],
    [46, 46, 46, 46],
    [49, 49, 49, 49],
    [40, 40, 40, 40, 54, 54, 54, 54, 54, 54, 54],
    [36, 36, 36, 36, 52, 52, 52, 52, 52, 52, 52],
    [35, 35, 35, 35, 54, 54, 54, 54, 54, 54, 54, 31, 31, 31, 31, 52, 52, 52, 52, 52, 52, 52],
  ]
  assert to_list(zo_layer_to_zo_bounds_lut_1) == zo_layer_to_zo_bounds_lut_1_answer

  # Further tests (zone 1)
  zone = 2
  num_emtf_zo_layers = 8
  ri_layer_to_zo_layer_lut = find_emtf_zo_layer_lut()[:, zone]
  zo_layer_to_ri_layer_lut = to_array(inverse_fn(ri_layer_to_zo_layer_lut, range(num_emtf_zo_layers)))

  # ME1/2, RE1/2, RE2/2, ME2/2, ME3/2, RE3/2, ME4/2, RE4/2
  zo_layer_to_ri_layer_lut_answer = [[1], [10], [13], [4], [6], [15], [8], [17]]
  assert to_list(zo_layer_to_ri_layer_lut) == zo_layer_to_ri_layer_lut_answer

  ri_layer_to_zo_bounds_lut_0 = find_emtf_zones_lut()[:, zone, 0]
  ri_layer_to_zo_bounds_lut_1 = find_emtf_zones_lut()[:, zone, 1]
  zo_layer_to_chamber_lut = to_array([
      [c for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])
  zo_layer_to_zo_bounds_lut_0 = to_array([
      [ri_layer_to_zo_bounds_lut_0[ri_layer] for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])
  zo_layer_to_zo_bounds_lut_1 = to_array([
      [ri_layer_to_zo_bounds_lut_1[ri_layer] for ri_layer in ri_layers for c in ri_layer_to_chamber_lut[ri_layer]] \
      for ri_layers in zo_layer_to_ri_layer_lut
  ])

  zo_layer_to_chamber_lut_answer = [
    [  3,   4,   5,  12,  13,  14,  46],
    [ 57,  58,  59,  66,  67,  68, 100],
    [ 75,  76,  77,  78,  79,  80, 103],
    [ 21,  22,  23,  24,  25,  26,  49],
    [ 30,  31,  32,  33,  34,  35,  51],
    [ 84,  85,  86,  87,  88,  89, 105],
    [ 39,  40,  41,  42,  43,  44,  53],
    [ 93,  94,  95,  96,  97,  98, 107],
  ]
  assert to_list(zo_layer_to_chamber_lut) == zo_layer_to_chamber_lut_answer

  zo_layer_to_zo_bounds_lut_0_answer = [
    [52, 52, 52, 52, 52, 52, 52],
    [52, 52, 52, 52, 52, 52, 52],
    [56, 56, 56, 56, 56, 56, 56],
    [53, 53, 53, 53, 53, 53, 53],
    [51, 51, 51, 51, 51, 51, 51],
    [48, 48, 48, 48, 48, 48, 48],
    [51, 51, 51, 51, 51, 51, 51],
    [52, 52, 52, 52, 52, 52, 52],
  ]
  assert to_list(zo_layer_to_zo_bounds_lut_0) == zo_layer_to_zo_bounds_lut_0_answer

  zo_layer_to_zo_bounds_lut_1_answer = [
    [88, 88, 88, 88, 88, 88, 88],
    [84, 84, 84, 84, 84, 84, 84],
    [88, 88, 88, 88, 88, 88, 88],
    [88, 88, 88, 88, 88, 88, 88],
    [88, 88, 88, 88, 88, 88, 88],
    [84, 84, 84, 84, 84, 84, 84],
    [88, 88, 88, 88, 88, 88, 88],
    [84, 84, 84, 84, 84, 84, 84],
  ]
  assert to_list(zo_layer_to_zo_bounds_lut_1) == zo_layer_to_zo_bounds_lut_1_answer
