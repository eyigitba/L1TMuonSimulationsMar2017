"""Data analysis (quick)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from emtf_algos import *


# ______________________________________________________________________________
# Analyses

class _BaseAnalysis(object):
  """Abstract base class"""
  pass

# ______________________________________________________________________________
class DummyAnalysis(_BaseAnalysis):
  """Dummy analysis.

  Description.
  """

  def run(self, algo):
    # Load tree
    tree = load_pgun_test()

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      if verbosity >= kINFO:
        print('evt {0} has {1} particles and {2} hits'.format(ievt, len(evt.particles), len(evt.hits)))

      # Particles
      part = evt.particles[0]  # particle gun
      if verbosity >= kINFO:
        print('.. part {0} {1} {2} {3} {4} {5}'.format(0, part.pt, part.eta, part.phi, part.invpt, part.d0))

      # Sim hits
      if verbosity >= kINFO:
        for isimhit, simhit in enumerate(evt.simhits):
          simhit_id = (simhit.type, simhit.layer, simhit.chamber)
          print('.. simhit {0} {1} {2} {3}'.format(isimhit, simhit_id, simhit.phi, simhit.theta))

      # Trigger primitives
      if verbosity >= kINFO:
        for ihit, hit in enumerate(evt.hits):
          hit_id = (hit.type, hit.station, hit.ring, calc_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
          hit_sim_tp = hit.sim_tp1
          if (hit.type == kCSC) and (hit_sim_tp != hit.sim_tp2):
            hit_sim_tp = -1
          print('.. hit {0} {1} {2} {3} {4} {5} {6}'.format(ihit, hit_id, hit.emtf_phi, hit.emtf_theta, hit.bend, hit.quality, hit_sim_tp))

    # End loop over events
    return

# ______________________________________________________________________________
class ZoneAnalysis(_BaseAnalysis):
  """Find zone boundaries.

  Description.
  """

  def run(self, algo, pileup=200):
    # Overwrite maxevents
    maxevents = -1

    # Overwrite eta bins
    eta_bins = (0.8, 1.2, 1.55, 1.98, 2.5)

    def find_particle_zone_quick(eta):
      ind = np.searchsorted(eta_bins, np.abs(eta))
      return (len(eta_bins)-1) - ind  # ind = (1,2,3,4) -> zone (3,2,1,0)

    nzones = len(eta_bins) - 1
    out_hits = []
    out_hits_metadata = {'type': 0, 'station': 1, 'ring': 2, 'zone': 3, 'emtf_theta': 4}

    # __________________________________________________________________________
    # Load tree
    tree = load_pgun_batch(jobid)

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      # Particles (pT > 4 GeV)
      part = evt.particles[0]  # particle gun
      if not (part.pt > 4):
        continue

      # Find particle zone
      zone = find_particle_zone_quick(part.eta)

      # Trigger primitives
      for ihit, hit in enumerate(evt.hits):
        if is_emtf_legit_hit(hit):
          out_hits.append([hit.type, hit.station, hit.ring, zone, hit.emtf_theta])

    # End loop over events

    # __________________________________________________________________________
    # Compute results
    out_hits = np.asarray(out_hits, dtype=np.int32)
    out_hits_type = out_hits[:, out_hits_metadata['type']]
    out_hits_station = out_hits[:, out_hits_metadata['station']]
    out_hits_ring = out_hits[:, out_hits_metadata['ring']]
    out_hits_zone = out_hits[:, out_hits_metadata['zone']]
    out_hits_emtf_theta = out_hits[:, out_hits_metadata['emtf_theta']]

    out_hits_ri_layers = find_emtf_ri_layer(out_hits_type, out_hits_station, out_hits_ring)

    for zone in range(nzones):
      sel = (out_hits_zone == zone)
      out_hits_ri_layers_sel = out_hits_ri_layers[sel]
      out_hits_emtf_theta_sel = out_hits_emtf_theta[sel]
      layers = np.unique(out_hits_ri_layers_sel)

      for lay in layers:
        sel1 = (out_hits_ri_layers_sel == lay)
        out_hits_emtf_theta_sel1 = out_hits_emtf_theta_sel[sel1]
        n = len(out_hits_emtf_theta_sel1)
        if n > 100:
          p = np.percentile(out_hits_emtf_theta_sel1, [1,2,2.5,3,97,97.5,98,99], overwrite_input=True)
          print(zone, '%03i' % lay, '%5i' % n, p[:4], p[4:])
    return


# ______________________________________________________________________________
# Main

import os, sys, datetime

# Algorithm (pick one)
algo = 'default'  # phase 2
#algo = 'run3'

# Analysis mode (pick one)
analysis = 'dummy'
#analysis = 'zone'

# Job id (pick an integer)
jobid = 0

# Max num of events (-1 means all events)
maxevents = 100

# Verbosity
verbosity = 1

# Condor or not
# if 'CONDOR_EXEC' is defined, overwrite the 3 arguments (algo, analysis, jobid)
use_condor = ('CONDOR_EXEC' in os.environ)
if use_condor:
  nargs = 3
  if len(sys.argv) != (nargs + 1):
    raise RuntimeError('Expect num of arguments: {0}'.format(nargs))
  os.environ['ROOTPY_GRIDMODE'] = 'true'
  algo = sys.argv[1]
  analysis = sys.argv[2]
  jobid = int(sys.argv[3])
  maxevents = -1
  verbosity = 0

# Decorator
def app_decorator(fn):
  def wrapper(*args, **kwargs):
    # Begin
    start_time = datetime.datetime.now()
    print('[INFO] Current time    : {0}'.format(start_time))
    print('[INFO] Using cmssw     : {0}'.format(os.environ['CMSSW_VERSION']))
    print('[INFO] Using condor    : {0}'.format(use_condor))
    print('[INFO] Using algo      : {0}'.format(algo))
    print('[INFO] Using analysis  : {0}'.format(analysis))
    print('[INFO] Using jobid     : {0}'.format(jobid))
    print('[INFO] Using maxevents : {0}'.format(maxevents))
    # Run
    fn(*args, **kwargs)
    # End
    stop_time = datetime.datetime.now()
    print('[INFO] Elapsed time    : {0}'.format(stop_time - start_time))
    return
  return wrapper

# App
@app_decorator
def app():
  # Select analysis
  if analysis == 'dummy':
    myapp = DummyAnalysis()
    myargs = dict(algo=algo)

  elif analysis == 'zone':
    myapp = ZoneAnalysis()
    myargs = dict(algo=algo)

  else:
    raise RuntimeError('Cannot recognize analysis: {0}'.format(analysis))

  # Run analysis
  myapp.run(**myargs)
  return

# Finally
if __name__ == '__main__':
  app()
