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
# Total: 12 (5 from CSC + 4 from RPC + 3 from GEM)
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
# Total: 19 (9*2 + 1)
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


# Encode EMTF chamber number
# Total: 112 (6*9*2 + 4)
def find_emtf_chamber_initializer():
  default_value = -99
  lut = np.full((5,5,10,4), default_value, dtype=np.int32)  # (type, station, cscid, subsector) -> chamber
  lut[1,1,1,1] = 0   # ME1/1 sub 1
  lut[1,1,2,1] = 1   # ME1/1 sub 1
  lut[1,1,3,1] = 2   # ME1/1 sub 1
  lut[1,1,4,1] = 3   # ME1/2 sub 1
  lut[1,1,5,1] = 4   # ME1/2 sub 1
  lut[1,1,6,1] = 5   # ME1/2 sub 1
  lut[1,1,7,1] = 6   # ME1/3 sub 1
  lut[1,1,8,1] = 7   # ME1/3 sub 1
  lut[1,1,9,1] = 8   # ME1/3 sub 1
  lut[1,1,1,2] = 9   # ME1/1 sub 2
  lut[1,1,2,2] = 10  # ME1/1 sub 2
  lut[1,1,3,2] = 11  # ME1/1 sub 2
  lut[1,1,4,2] = 12  # ME1/2 sub 2
  lut[1,1,5,2] = 13  # ME1/2 sub 2
  lut[1,1,6,2] = 14  # ME1/2 sub 2
  lut[1,1,7,2] = 15  # ME1/3 sub 2
  lut[1,1,8,2] = 16  # ME1/3 sub 2
  lut[1,1,9,2] = 17  # ME1/3 sub 2
  lut[1,2,1,0] = 18  # ME2/1
  lut[1,2,2,0] = 19  # ME2/1
  lut[1,2,3,0] = 20  # ME2/1
  lut[1,2,4,0] = 21  # ME2/2
  lut[1,2,5,0] = 22  # ME2/2
  lut[1,2,6,0] = 23  # ME2/2
  lut[1,2,7,0] = 24  # ME2/2
  lut[1,2,8,0] = 25  # ME2/2
  lut[1,2,9,0] = 26  # ME2/2
  lut[1,3,1,0] = 27  # ME3/1
  lut[1,3,2,0] = 28  # ME3/1
  lut[1,3,3,0] = 29  # ME3/1
  lut[1,3,4,0] = 30  # ME3/2
  lut[1,3,5,0] = 31  # ME3/2
  lut[1,3,6,0] = 32  # ME3/2
  lut[1,3,7,0] = 33  # ME3/2
  lut[1,3,8,0] = 34  # ME3/2
  lut[1,3,9,0] = 35  # ME3/2
  lut[1,4,1,0] = 36  # ME4/1
  lut[1,4,2,0] = 37  # ME4/1
  lut[1,4,3,0] = 38  # ME4/1
  lut[1,4,4,0] = 39  # ME4/2
  lut[1,4,5,0] = 40  # ME4/2
  lut[1,4,6,0] = 41  # ME4/2
  lut[1,4,7,0] = 42  # ME4/2
  lut[1,4,8,0] = 43  # ME4/2
  lut[1,4,9,0] = 44  # ME4/2
  lut[1,1,3,3] = 45  # ME1/1 neigh
  lut[1,1,6,3] = 46  # ME1/2 neigh
  lut[1,1,9,3] = 47  # ME1/3 neigh
  lut[1,2,3,3] = 48  # ME2/1 neigh
  lut[1,2,9,3] = 49  # ME2/2 neigh
  lut[1,3,3,3] = 50  # ME3/1 neigh
  lut[1,3,9,3] = 51  # ME3/2 neigh
  lut[1,4,3,3] = 52  # ME4/1 neigh
  lut[1,4,9,3] = 53  # ME4/2 neigh
  #
  lut[3,1,1,1] = 54  # GE1/1 sub 1
  lut[3,1,2,1] = 55  # GE1/1 sub 1
  lut[3,1,3,1] = 56  # GE1/1 sub 1
  lut[2,1,4,1] = 57  # RE1/2 sub 1
  lut[2,1,5,1] = 58  # RE1/2 sub 1
  lut[2,1,6,1] = 59  # RE1/2 sub 1
  lut[2,1,7,1] = 60  # RE1/3 sub 1
  lut[2,1,8,1] = 61  # RE1/3 sub 1
  lut[2,1,9,1] = 62  # RE1/3 sub 1
  lut[3,1,1,2] = 63  # GE1/1 sub 2
  lut[3,1,2,2] = 64  # GE1/1 sub 2
  lut[3,1,3,2] = 65  # GE1/1 sub 2
  lut[2,1,4,2] = 66  # RE1/2 sub 2
  lut[2,1,5,2] = 67  # RE1/2 sub 2
  lut[2,1,6,2] = 68  # RE1/2 sub 2
  lut[2,1,7,2] = 69  # RE1/3 sub 2
  lut[2,1,8,2] = 70  # RE1/3 sub 2
  lut[2,1,9,2] = 71  # RE1/3 sub 2
  lut[3,2,1,0] = 72  # GE2/1
  lut[3,2,2,0] = 73  # GE2/1
  lut[3,2,3,0] = 74  # GE2/1
  lut[2,2,4,0] = 75  # RE2/2
  lut[2,2,5,0] = 76  # RE2/2
  lut[2,2,6,0] = 77  # RE2/2
  lut[2,2,7,0] = 78  # RE2/2
  lut[2,2,8,0] = 79  # RE2/2
  lut[2,2,9,0] = 80  # RE2/2
  lut[2,3,1,0] = 81  # RE3/1
  lut[2,3,2,0] = 82  # RE3/1
  lut[2,3,3,0] = 83  # RE3/1
  lut[2,3,4,0] = 84  # RE3/2
  lut[2,3,5,0] = 85  # RE3/2
  lut[2,3,6,0] = 86  # RE3/2
  lut[2,3,7,0] = 87  # RE3/2
  lut[2,3,8,0] = 88  # RE3/2
  lut[2,3,9,0] = 89  # RE3/2
  lut[2,4,1,0] = 90  # RE4/1
  lut[2,4,2,0] = 91  # RE4/1
  lut[2,4,3,0] = 92  # RE4/1
  lut[2,4,4,0] = 93  # RE4/2
  lut[2,4,5,0] = 94  # RE4/2
  lut[2,4,6,0] = 95  # RE4/2
  lut[2,4,7,0] = 96  # RE4/2
  lut[2,4,8,0] = 97  # RE4/2
  lut[2,4,9,0] = 98  # RE4/2
  lut[3,1,3,3] = 99  # GE1/1 neigh
  lut[2,1,6,3] = 100 # RE1/2 neigh
  lut[2,1,9,3] = 101 # RE1/3 neigh
  lut[3,2,3,3] = 102 # GE2/1 neigh
  lut[2,2,9,3] = 103 # RE2/2 neigh
  lut[2,3,3,3] = 104 # RE3/1 neigh
  lut[2,3,9,3] = 105 # RE3/2 neigh
  lut[2,4,3,3] = 106 # RE4/1 neigh
  lut[2,4,9,3] = 107 # RE4/2 neigh
  #
  lut[4,1,1,0] = 108 # ME0
  lut[4,1,2,0] = 109 # ME0
  lut[4,1,3,0] = 110 # ME0
  lut[4,1,1,3] = 111 # ME0 neigh
  lut[4,1,2,3] = 111 # ME0 neigh
  lut[4,1,3,3] = 111 # ME0 neigh

  def lookup(_type, station, cscid, subsector, neighbor):
    subsector = np.where(neighbor, 3, subsector)  # neighbor -> subsector 3
    multi_index = np.array([_type, station, cscid, subsector])
    flat_index = np.ravel_multi_index(multi_index, lut.shape)
    item = np.take(lut, flat_index)
    return item
  return lookup

# The initializer will instantiate the lookup table
find_emtf_chamber = find_emtf_chamber_initializer()

# Decode EMTF chamber number
def decode_emtf_chamber():
  raise NotImplementedError()


# Assign EMTF zones
def find_emtf_zones_initializer():
  default_value = -99
  lut = np.full((19,3,2), default_value, dtype=np.int32)  # (ri_layer,zone) -> (min_theta,max_theta)
  lut[0,0] = 4,26    # ME1/1
  lut[3,0] = 4,25    # ME2/1
  lut[5,0] = 4,25    # ME3/1
  lut[7,0] = 4,25    # ME4/1
  lut[9,0] = 17,26   # GE1/1
  lut[12,0] = 4,25   # GE2/1
  lut[14,0] = 4,25   # RE3/1
  lut[16,0] = 4,25   # RE4/1
  lut[18,0] = 4,23   # ME0
  #
  lut[0,1] = 24,53   # ME1/1
  lut[1,1] = 46,54   # ME1/2
  lut[3,1] = 23,49   # ME2/1
  lut[5,1] = 23,40   # ME3/1
  lut[6,1] = 44,54   # ME3/2
  lut[7,1] = 23,35   # ME4/1
  lut[8,1] = 38,54   # ME4/2
  lut[9,1] = 24,52   # GE1/1
  lut[10,1] = 52,56  # RE1/2
  lut[12,1] = 23,46  # GE2/1
  lut[14,1] = 23,36  # RE3/1
  lut[15,1] = 40,52  # RE3/2
  lut[16,1] = 23,31  # RE4/1
  lut[17,1] = 36,52  # RE4/2
  #
  lut[1,2] = 52,88   # ME1/2
  lut[4,2] = 53,88   # ME2/2
  lut[6,2] = 51,88   # ME3/2
  lut[8,2] = 51,88   # ME4/2
  lut[10,2] = 52,84  # RE1/2
  lut[13,2] = 56,88  # RE2/2
  lut[15,2] = 48,84  # RE3/2
  lut[17,2] = 52,84  # RE4/2

  def lookup(ri_layer, emtf_theta):
    ri_layer = np.asarray(ri_layer)
    emtf_theta = np.asarray(emtf_theta)
    bounds = np.take(lut, ri_layer, axis=0)
    # Create a boolean array representing the bits in a uint8 array. Set the last 3 bits.
    # Then, pack into a uint8 array
    if ri_layer.ndim == 0:
      result = np.zeros(8, dtype=np.bool)
      result[-3:] = (bounds[..., 0] <= emtf_theta) & (emtf_theta <= bounds[..., 1])
      result = np.packbits(result)
    else:
      result = np.zeros(ri_layer.shape + (8,), dtype=np.bool)
      result[..., -3:] = (bounds[..., 0] <= emtf_theta[:, np.newaxis]) & (emtf_theta[:, np.newaxis] <= bounds[..., 1])
      result = np.packbits(result)
    return result
  return lookup

# The initializer will instantiate the lookup table
find_emtf_zones = find_emtf_zones_initializer()

# Assign EMTF timezones
def find_emtf_timezones_initializer():
  default_value = -99
  lut = np.full((19,3,2), default_value, dtype=np.int32)  # (ri_layer,timezone) -> (min_bx,max_bx)
  lut[0,1] = -1,0    # ME1/1
  lut[1,1] = -1,0    # ME1/2
  lut[2,1] = -1,0    # ME1/3
  lut[3,1] = -1,0    # ME2/1
  lut[4,1] = -1,0    # ME2/2
  lut[5,1] = -1,0    # ME3/1
  lut[6,1] = -1,0    # ME3/2
  lut[7,1] = -1,0    # ME4/1
  lut[8,1] = -1,0    # ME4/2
  lut[9,1] = 0,0     # GE1/1
  lut[10,1] = 0,0    # RE1/2
  lut[11,1] = 0,0    # RE1/3
  lut[12,1] = 0,0    # GE2/1
  lut[13,1] = 0,0    # RE2/2
  lut[14,1] = 0,0    # RE3/1
  lut[15,1] = 0,0    # RE3/2
  lut[16,1] = 0,0    # RE4/1
  lut[17,1] = 0,0    # RE4/2
  lut[18,1] = 0,0    # ME0
  #
  lut[:, 0] = lut[:, 1] - 1  # timezone 0 = timezone 1 - 1
  lut[:, 2] = lut[:, 1] + 1  # timezone 2 = timezone 1 + 1

  def lookup(ri_layer, bx):
    ri_layer = np.asarray(ri_layer)
    bx = np.asarray(bx)
    bounds = np.take(lut, ri_layer, axis=0)
    # Create a boolean array representing the bits in a uint8 array. Set the last 3 bits.
    # Then, pack into a uint8 array
    if ri_layer.ndim == 0:
      result = np.zeros(8, dtype=np.bool)
      result[-3:] = (bounds[..., 0] <= bx) & (bx <= bounds[..., 1])
      result = np.packbits(result)
    else:
      result = np.zeros(ri_layer.shape + (8,), dtype=np.bool)
      result[..., -3:] = (bounds[..., 0] <= bx[:, np.newaxis]) & (bx[:, np.newaxis] <= bounds[..., 1])
      result = np.packbits(result)
    return result
  return lookup

# The initializer will instantiate the lookup table
find_emtf_timezones = find_emtf_timezones_initializer()

# Encode EMTF zone image layer
def find_emtf_zo_layer_initializer():
  # Layer ordering (closest to furthest):
  # ME0, GE1/1, ME1/1,
  # ME1/2, RE1,
  # GE2/1, RE2, ME2,
  # ME3, RE3,
  # ME4, RE4,
  default_value = -99
  lut = np.full((19,3), default_value, dtype=np.int32)  # (ri_layer,zone) -> zo_layer
  lut[0,0] = 2   # ME1/1
  lut[3,0] = 4   # ME2/1
  lut[5,0] = 5   # ME3/1
  lut[7,0] = 7   # ME4/1
  lut[9,0] = 1   # GE1/1
  lut[12,0] = 3  # GE2/1
  lut[14,0] = 6  # RE3/1
  lut[16,0] = 7  # RE4/1
  lut[18,0] = 0  # ME0
  #
  lut[0,1] = 1   # ME1/1
  lut[1,1] = 2   # ME1/2
  lut[3,1] = 4   # ME2/1
  lut[5,1] = 5   # ME3/1
  lut[6,1] = 5   # ME3/2
  lut[7,1] = 7   # ME4/1
  lut[8,1] = 7   # ME4/2
  lut[9,1] = 0   # GE1/1
  lut[10,1] = 2  # RE1/2
  lut[12,1] = 3  # GE2/1
  lut[14,1] = 6  # RE3/1
  lut[15,1] = 6  # RE3/2
  lut[16,1] = 7  # RE4/1
  lut[17,1] = 7  # RE4/2
  #
  lut[1,2] = 0   # ME1/2
  lut[4,2] = 3   # ME2/2
  lut[6,2] = 4   # ME3/2
  lut[8,2] = 6   # ME4/2
  lut[10,2] = 1  # RE1/2
  lut[13,2] = 2  # RE2/2
  lut[15,2] = 5  # RE3/2
  lut[17,2] = 7  # RE4/2

  def lookup(ri_layer, zone):
    ri_layer = np.asarray(ri_layer)
    zone = np.asarray(zone)
    if zone.ndim != 0:
      raise ValueError('zone should be a scalar.')
    item = np.take(lut[:, zone], ri_layer)
    return item
  return lookup

# The initializer will instantiate the lookup table
find_emtf_zo_layer = find_emtf_zo_layer_initializer()


# Check whether hit is very legal and very cool
def is_emtf_legit_hit(hit):
  def check_type(hit):
    is_dt = (hit.type == kDT)
    return (not is_dt)
  def check_phi(hit):
    if hasattr(hit, 'emtf_phi'):
      if hit.type == kME0 or hit.type == kDT:
        return hit.emtf_phi > 0
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
    if hasattr(hit, 'emtf_phi'):
      if hit.type == kME0 or hit.type == kDT:
        return hit.emtf_phi > 0
    return True
  return check_type(hit) and check_phi(hit)
