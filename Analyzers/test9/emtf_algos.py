"""Algorithms in EMTF++."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from emtf_utils import *


# ______________________________________________________________________________
# Classes

num_emtf_sectors = 12

min_emtf_strip = 5*64    # 5 deg

max_emtf_strip = 80*64   # 80 deg

coarse_emtf_strip = 8*2  # 'doublestrip' unit

emtf_eta_bins = (0.8, 1.2, 1.55, 1.98, 2.5)

# Encode EMTF layer number
def find_emtf_layer_initializer():
  default_value = -99
  lut = np.full((5,5,5), default_value, dtype=np.int32)  # (type, station, ring) -> layer
  lut[1,1,4] = 0  # ME1/1a
  lut[1,1,1] = 0  # ME1/1b
  lut[1,1,2] = 1  # ME1/2
  lut[1,1,3] = 1  # ME1/3
  lut[1,2,1] = 2  # ME2/1
  lut[1,2,2] = 2  # ME2/2
  lut[1,3,1] = 3  # ME3/1
  lut[1,3,2] = 3  # ME3/2
  lut[1,4,1] = 4  # ME4/1
  lut[1,4,2] = 4  # ME4/2
  lut[2,1,2] = 5  # RE1/2
  lut[2,1,3] = 5  # RE1/3
  lut[2,2,2] = 6  # RE2/2
  lut[2,2,3] = 6  # RE2/3
  lut[2,3,1] = 7  # RE3/1
  lut[2,3,2] = 7  # RE3/2
  lut[2,3,3] = 7  # RE3/3
  lut[2,4,1] = 8  # RE4/1
  lut[2,4,2] = 8  # RE4/2
  lut[2,4,3] = 8  # RE4/3
  lut[3,1,1] = 9  # GE1/1
  lut[3,2,1] = 10 # GE2/1
  lut[4,1,1] = 11 # ME0

  def lookup(_type, station, ring):
    multi_index = np.array([_type, station, ring])
    flat_index = np.ravel_multi_index(multi_index, lut.shape)
    item = np.take(lut, flat_index)
    return item
  return lookup

# The initializer will instantiate the lookup table
find_emtf_layer = find_emtf_layer_initializer()

# Encode EMTF ri_layer number that retains some ring info
def find_emtf_ri_layer_initializer():
  default_value = -99
  lut = np.full((5,5,5), default_value, dtype=np.int32)  # (type, station, ring) -> layer
  lut[1,1,4] = 0  # ME1/1a
  lut[1,1,1] = 0  # ME1/1b
  lut[1,1,2] = 1  # ME1/2
  lut[1,1,3] = 2  # ME1/3
  lut[1,2,1] = 3  # ME2/1
  lut[1,2,2] = 4  # ME2/2
  lut[1,3,1] = 5  # ME3/1
  lut[1,3,2] = 6  # ME3/2
  lut[1,4,1] = 7  # ME4/1
  lut[1,4,2] = 8  # ME4/2
  lut[3,1,1] = 9  # GE1/1
  lut[2,1,2] = 10 # RE1/2
  lut[2,1,3] = 11 # RE1/3
  lut[3,2,1] = 12 # GE2/1
  lut[2,2,2] = 13 # RE2/2
  lut[2,2,3] = 13 # RE2/3
  lut[2,3,1] = 14 # RE3/1
  lut[2,3,2] = 15 # RE3/2
  lut[2,3,3] = 15 # RE3/3
  lut[2,4,1] = 16 # RE4/1
  lut[2,4,2] = 17 # RE4/2
  lut[2,4,3] = 17 # RE4/3
  lut[4,1,1] = 18 # ME0

  def lookup(_type, station, ring):
    multi_index = np.array([_type, station, ring])
    flat_index = np.ravel_multi_index(multi_index, lut.shape)
    item = np.take(lut, flat_index)
    return item
  return lookup

# The initializer will instantiate the lookup table
find_emtf_ri_layer = find_emtf_ri_layer_initializer()

# Decode EMTF layer number
def decode_emtf_layer_initializer():
  lut_0 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4])  # exact
  lut_1 = np.array([1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1])  # exact
  lut_2 = np.array([1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1])  # inexact

  def lookup(emtf_layer):
    _type = np.take(lut_0, emtf_layer)
    station = np.take(lut_1, emtf_layer)
    ring = np.take(lut_2, emtf_layer)
    return (_type, station, ring)
  return lookup

# The initializer will instantiate the lookup tables
decode_emtf_layer = decode_emtf_layer_initializer()

# Decode EMTF ri_layer number that retains some ring info
def decode_emtf_ri_layer_initializer():
  lut_0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 3, 2, 2, 2, 2, 2, 4])  # exact
  lut_1 = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 2, 2, 3, 3, 4, 4, 1])  # exact
  lut_2 = np.array([1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1])  # inexact

  def lookup(emtf_layer):
    _type = np.take(lut_0, emtf_layer)
    station = np.take(lut_1, emtf_layer)
    ring = np.take(lut_2, emtf_layer)
    return (_type, station, ring)
  return lookup

# The initializer will instantiate the lookup tables
decode_emtf_ri_layer = decode_emtf_ri_layer_initializer()


# Check whether hit is very legal and very cool
def is_emtf_legit_hit(hit):
  def check_type(hit):
    is_dt = (hit.type == kDT)
    return (not is_dt)
  def check_phi(hit):
    if hit.type == kME0 or hit.type == kDT:
      return hit.emtf_phi > 0
    else:
      return True
  return check_type(hit) and check_phi(hit)

# Check whether hit is very legal and very cool for Run 2
def is_emtf_legit_hit_run2(hit):
  def check_type(hit):
    is_csc = (hit.type == kCSC)
    is_rpc_run2 = (hit.type == kRPC) \
        and not ((hit.station == 3 or hit.station == 4) and hit.ring == 1) \
        and not ((hit.station == 1 or hit.station == 2) and hit.ring == 3)
    return (is_csc or is_rpc_run2)
  def check_phi(hit):
    if hit.type == kME0 or hit.type == kDT:
      return hit.emtf_phi > 0
    else:
      return True
  return check_type(hit) and check_phi(hit)
