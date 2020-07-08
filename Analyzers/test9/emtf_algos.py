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

# Encode EMTF hit layer number (Y16)
class EMTFLayerY16(object):
  def __init__(self):
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
    lut[0,1,1] = 12 # MB1
    lut[0,2,1] = 13 # MB2
    lut[0,3,1] = 14 # MB3
    lut[0,4,1] = 15 # MB4
    self.lut = lut

  def __call__(self, hit):
    index = (hit.type, hit.station, hit.ring)
    emtf_layer = self.lut[index]
    return emtf_layer

# Encode EMTF hit layer number (Y27)
class EMTFLayerY27(object):
  def __init__(self):
    default_value = -99
    lut = np.full((5,5,5), default_value, dtype=np.int32)  # (type, station, ring) -> layer
    lut[1,1,4] = 0  # ME1/1a
    lut[1,1,1] = 1  # ME1/1b
    lut[1,1,2] = 2  # ME1/2
    lut[1,1,3] = 3  # ME1/3
    lut[1,2,1] = 4  # ME2/1
    lut[1,2,2] = 5  # ME2/2
    lut[1,3,1] = 6  # ME3/1
    lut[1,3,2] = 7  # ME3/2
    lut[1,4,1] = 8  # ME4/1
    lut[1,4,2] = 9  # ME4/2
    lut[2,1,2] = 10 # RE1/2
    lut[2,1,3] = 11 # RE1/3
    lut[2,2,2] = 12 # RE2/2
    lut[2,2,3] = 13 # RE2/3
    lut[2,3,1] = 14 # RE3/1
    lut[2,3,2] = 15 # RE3/2
    lut[2,3,3] = 16 # RE3/3
    lut[2,4,1] = 17 # RE4/1
    lut[2,4,2] = 18 # RE4/2
    lut[2,4,3] = 19 # RE4/3
    lut[3,1,1] = 20 # GE1/1
    lut[3,2,1] = 21 # GE2/1
    lut[4,1,1] = 22 # ME0
    lut[0,1,1] = 23 # MB1
    lut[0,2,1] = 24 # MB2
    lut[0,3,1] = 25 # MB3
    lut[0,4,1] = 26 # MB4
    self.lut = lut

  def __call__(self, hit):
    index = (hit.type, hit.station, hit.ring)
    emtf_layer = self.lut[index]
    return emtf_layer

# Decode EMTF hit layer number (Y16)
class EMTFLayerInverseY16(object):
  def __init__(self):
    lut = [
      (1,1,1), (1,1,2), (1,2,1), (1,3,1), (1,4,1), (2,1,1), (2,2,1), (2,3,1), (2,4,1),
      (3,1,1), (3,2,1), (4,1,1), (0,1,1), (0,2,1), (0,3,1), (0,4,1),
    ]
    lut = np.asarray(lut, dtype=np.int32)
    self.lut = lut

  def __call__(self, value):
    (_type, station, ring) = self.lut[value]
    return (_type, station, ring)

# Decode EMTF hit layer number (Y27)
class EMTFLayerInverseY27(object):
  def __init__(self):
    lut = [
      (1,1,4), (1,1,1), (1,1,2), (1,1,3), (1,2,1), (1,2,2), (1,3,1), (1,3,2), (1,4,1), (1,4,2),
      (2,1,2), (2,1,3), (2,2,2), (2,2,3), (2,3,1), (2,3,2), (2,3,3), (2,4,1), (2,4,2), (2,4,3),
      (3,1,1), (3,2,1), (4,1,1), (0,1,1), (0,2,1), (0,3,1), (0,4,1),
    ]
    lut = np.asarray(lut, dtype=np.int32)
    self.lut = lut

  def __call__(self, value):
    (_type, station, ring) = self.lut[value]
    return (_type, station, ring)

# Functionalize
find_emtf_layer = EMTFLayerY16()
find_emtf_layer_Y27 = EMTFLayerY27()
decode_emtf_layer = EMTFLayerInverseY16()
decode_emtf_layer_Y27 = EMTFLayerInverseY27()
