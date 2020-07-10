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

from emtf_utils import *


# ______________________________________________________________________________
def test_wrap_phi_rad():
  assert wrap_phi_rad(0.5) == pytest.approx(0.5)
  assert wrap_phi_rad(0.5 + np.pi*2) == pytest.approx(0.5)
  assert wrap_phi_rad(0.5 - np.pi*2) == pytest.approx(0.5)
  assert wrap_phi_rad(0.5 + np.pi) == pytest.approx(0.5 - np.pi)
  assert wrap_phi_rad(-0.5 - np.pi) == pytest.approx(-0.5 + np.pi)

def test_wrap_phi_deg():
  assert wrap_phi_deg(60.) == pytest.approx(60.)
  assert wrap_phi_deg(60. + 180.*2) == pytest.approx(60.)
  assert wrap_phi_deg(60. - 180.*2) == pytest.approx(60.)
  assert wrap_phi_deg(60. + 180.) == pytest.approx(60. - 180.)
  assert wrap_phi_deg(-60. - 180.) == pytest.approx(-60. + 180.)

def test_wrap_theta_rad():
  assert wrap_theta_rad(np.pi) == pytest.approx(0.)
  assert wrap_theta_rad(-np.pi) == pytest.approx(0.)
  assert wrap_theta_rad(np.pi/2) == pytest.approx(np.pi/2)
  assert wrap_theta_rad(-np.pi/2) == pytest.approx(np.pi/2)
  assert wrap_theta_rad(np.pi/4) == pytest.approx(np.pi/4)
  assert wrap_theta_rad(-np.pi/4) == pytest.approx(np.pi/4)
  assert wrap_theta_rad(np.pi*3/4) == pytest.approx(np.pi/4)
  assert wrap_theta_rad(-np.pi*3/4) == pytest.approx(np.pi/4)

def test_wrap_theta_deg():
  assert wrap_theta_deg(180.) == pytest.approx(0.)
  assert wrap_theta_deg(-180.) == pytest.approx(0.)
  assert wrap_theta_deg(90.) == pytest.approx(90.)
  assert wrap_theta_deg(-90.) == pytest.approx(90.)
  assert wrap_theta_deg(45.) == pytest.approx(45.)
  assert wrap_theta_deg(-45.) == pytest.approx(45.)
  assert wrap_theta_deg(135.) == pytest.approx(45.)
  assert wrap_theta_deg(-135.) == pytest.approx(45.)
