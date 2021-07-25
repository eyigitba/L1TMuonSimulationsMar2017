"""Data exploration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from numba import njit

from emtf_algos import *
from emtf_ntuples import *


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
          hit_id = (hit.type, hit.station, hit.ring, get_trigger_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
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

  def run(self, algo):
    # Overwrite maxevents
    maxevents = -1

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
      zone = find_particle_zone(part.eta)

      # Trigger primitives
      for ihit, hit in enumerate(evt.hits):
        if is_emtf_legit_hit(hit):
          out_hits.append([hit.type, hit.station, hit.ring, zone, hit.emtf_theta])

    # End loop over events
    out_hits = np.asarray(out_hits, dtype=np.int32)

    # __________________________________________________________________________
    # Compute results
    out_hits_type = out_hits[:, out_hits_metadata['type']]
    out_hits_station = out_hits[:, out_hits_metadata['station']]
    out_hits_ring = out_hits[:, out_hits_metadata['ring']]
    out_hits_zone = out_hits[:, out_hits_metadata['zone']]
    out_hits_emtf_theta = out_hits[:, out_hits_metadata['emtf_theta']]

    out_hits_emtf_host = find_emtf_host(out_hits_type, out_hits_station, out_hits_ring)
    assert (out_hits_emtf_host != -99).all()

    print('Find zone boundaries')
    n_zones = len(emtf_eta_bins) - 1
    for zone in range(n_zones):
      sel = (out_hits_zone == zone)
      out_hits_emtf_host_sel = out_hits_emtf_host[sel]
      out_hits_emtf_theta_sel = out_hits_emtf_theta[sel]
      out_hits_emtf_host_uniq = np.unique(out_hits_emtf_host_sel)

      for emtf_host in out_hits_emtf_host_uniq:
        sel_1 = (out_hits_emtf_host_sel == emtf_host)
        out_hits_emtf_theta_sel_1 = out_hits_emtf_theta_sel[sel_1]
        n = len(out_hits_emtf_theta_sel_1)
        if n > 100:
          p = np.percentile(out_hits_emtf_theta_sel_1, [1,2,2.5,3,97,97.5,98,99], overwrite_input=True)
          print(zone, '%03i' % emtf_host, '%5i' % n, p[:4], p[4:])

    # Done
    return

# ______________________________________________________________________________
class ChamberAnalysis(_BaseAnalysis):
  """Check chamber num of segments.

  Description.
  """

  def run(self, algo):
    # Overwrite maxevents
    maxevents = -1

    out_hits = []
    out_hits_metadata = {'type': 0, 'station': 1, 'sector': 2, 'subsector': 3, 'cscid': 4, 'neighbor': 5, 'bx': 6, 'ievt': 7}

    out_simhits = []
    out_simhits_metadata = {'type': 0, 'station': 1, 'sector': 2, 'subsector': 3, 'cscid': 4, 'neighbor': 5, 'bx': 6, 'ievt': 7}

    # __________________________________________________________________________
    # Load tree
    tree = load_pgun_batch(jobid)

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      # Trigger primitives
      for ihit, hit in enumerate(evt.hits):
        if is_emtf_legit_hit(hit):
          if hit.type == kME0:
            # Special case for ME0 as it is a 20-deg chamber in station 1
            hack_me0_hit_chamber(hit)
          out_hits.append([hit.type, hit.station, get_trigger_endsec(hit.endcap, hit.sector), hit.subsector, hit.cscid, hit.neighbor, hit.bx, ievt])

      # Sim hits
      for isimhit, simhit in enumerate(evt.simhits):
        simhit.endcap = +1 if simhit.z >= 0 else -1
        if is_emtf_legit_hit(simhit):
          if simhit.type == kME0:
            # Special case for ME0 as it is a 20-deg chamber in station 1
            hack_me0_hit_chamber(simhit)

          simhit.sector = get_trigger_sector(simhit.ring, simhit.station, simhit.chamber)
          simhit.subsector = get_trigger_subsector(simhit.ring, simhit.station, simhit.chamber)
          simhit.cscid = get_trigger_cscid(simhit.ring, simhit.station, simhit.chamber)
          simhit.neighid = get_trigger_neighid(simhit.ring, simhit.station, simhit.chamber)
          simhit.bx = 0
          simhit.neighbor = 0
          out_simhits.append([simhit.type, simhit.station, get_trigger_endsec(simhit.endcap, simhit.sector), simhit.subsector, simhit.cscid, simhit.neighbor, simhit.bx, ievt])

          # If neighbor, share simhit with the neighbor sector
          if simhit.neighid == 1:
            get_next_sector = lambda sector: (sector + 1) if sector != 6 else (sector + 1 - 6)
            simhit.neighbor = 1
            simhit.sector = get_next_sector(simhit.sector)
            out_simhits.append([simhit.type, simhit.station, get_trigger_endsec(simhit.endcap, simhit.sector), simhit.subsector, simhit.cscid, simhit.neighbor, simhit.bx, ievt])

    # End loop over events
    out_hits = np.asarray(out_hits, dtype=np.int32)
    out_simhits = np.asarray(out_simhits, dtype=np.int32)

    # __________________________________________________________________________
    # Compute results (1)
    out_hits_type = out_hits[:, out_hits_metadata['type']]
    out_hits_station = out_hits[:, out_hits_metadata['station']]
    out_hits_sector = out_hits[:, out_hits_metadata['sector']]
    out_hits_cscid = out_hits[:, out_hits_metadata['cscid']]
    out_hits_subsector = out_hits[:, out_hits_metadata['subsector']]
    out_hits_neighbor = out_hits[:, out_hits_metadata['neighbor']]
    out_hits_bx = out_hits[:, out_hits_metadata['bx']]
    out_hits_ievt = out_hits[:, out_hits_metadata['ievt']]

    out_hits_chambers = find_emtf_chamber(out_hits_type, out_hits_station, out_hits_cscid, out_hits_subsector, out_hits_neighbor)
    assert (out_hits_chambers != -99).all()

    sel = (out_hits_bx == 0) & (out_hits_sector == 0)  # check only one bx and one sector
    out_hits_chambers_sel = out_hits_chambers[sel]
    out_hits_ievt_sel = out_hits_ievt[sel]

    n_chambers = out_hits_chambers_sel.max() + 1
    n_events = out_hits_ievt_sel.max() + 1
    chamber_counts = np.zeros((n_chambers, n_events), dtype=np.int32)

    @njit
    def jit_op(out_hits_chambers_sel, out_hits_ievt_sel, chamber_counts):
      for i in range(len(out_hits_chambers_sel)):
        chamb = out_hits_chambers_sel[i]
        ievt = out_hits_ievt_sel[i]
        chamber_counts[chamb, ievt] += 1
      return

    jit_op(out_hits_chambers_sel, out_hits_ievt_sel, chamber_counts)

    print('Check chamber num of segments')
    chamber_counts_max = np.max(chamber_counts, axis=-1)
    for chamb in range(n_chambers):
      print(chamb, chamber_counts_max[chamb])

    # __________________________________________________________________________
    # Compute results (2)
    out_hits_type = out_simhits[:, out_simhits_metadata['type']]
    out_hits_station = out_simhits[:, out_simhits_metadata['station']]
    out_hits_sector = out_simhits[:, out_simhits_metadata['sector']]
    out_hits_cscid = out_simhits[:, out_simhits_metadata['cscid']]
    out_hits_subsector = out_simhits[:, out_simhits_metadata['subsector']]
    out_hits_neighbor = out_simhits[:, out_simhits_metadata['neighbor']]
    out_hits_bx = out_simhits[:, out_simhits_metadata['bx']]
    out_hits_ievt = out_simhits[:, out_simhits_metadata['ievt']]

    out_hits_chambers = find_emtf_chamber(out_hits_type, out_hits_station, out_hits_cscid, out_hits_subsector, out_hits_neighbor)
    assert (out_hits_chambers != -99).all()

    sel = (out_hits_bx == 0) & (out_hits_sector == 0)  # check only one bx and one sector
    out_hits_chambers_sel = out_hits_chambers[sel]
    out_hits_ievt_sel = out_hits_ievt[sel]

    n_chambers = out_hits_chambers_sel.max() + 1
    n_events = out_hits_ievt_sel.max() + 1
    chamber_counts = np.zeros((n_chambers, n_events), dtype=np.int32)

    jit_op(out_hits_chambers_sel, out_hits_ievt_sel, chamber_counts)

    print('Check chamber num of simhits')
    chamber_counts_max = np.max(chamber_counts, axis=-1)
    for chamb in range(n_chambers):
      print(chamb, chamber_counts_max[chamb])

    # Done
    return


# ______________________________________________________________________________
# Main

import os, sys, datetime

# Algorithm (pick one)
algo = 'default'  # phase 2
#algo = 'run3'

# Analysis mode (pick one)
#analysis = 'dummy'
#analysis = 'zone'
analysis = 'chamber'

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
    raise RuntimeError('Expect num of arguments: {}'.format(nargs))
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
    print('[INFO] Current time    : {}'.format(start_time))
    print('[INFO] Using cmssw     : {}'.format(os.environ['CMSSW_VERSION']))
    print('[INFO] Using condor    : {}'.format(use_condor))
    print('[INFO] Using algo      : {}'.format(algo))
    print('[INFO] Using analysis  : {}'.format(analysis))
    print('[INFO] Using jobid     : {}'.format(jobid))
    print('[INFO] Using maxevents : {}'.format(maxevents))
    # Run
    fn(*args, **kwargs)
    # End
    stop_time = datetime.datetime.now()
    print('[INFO] Elapsed time    : {}'.format(stop_time - start_time))
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

  elif analysis == 'chamber':
    myapp = ChamberAnalysis()
    myargs = dict(algo=algo)

  else:
    raise RuntimeError('Cannot recognize analysis: {}'.format(analysis))

  # Run analysis
  myapp.run(**myargs)
  return

# Finally
if __name__ == '__main__':
  app()
