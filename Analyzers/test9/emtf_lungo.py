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
    verbosity = 1

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
          hit_id = (hit.type, hit.station, hit.ring, find_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
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
    # Overwrite eta bins
    eta_bins = (0.8, 1.2, 1.55, 1.98, 2.5)
    eta_bins = eta_bins[::-1]

    def find_eta_bin(eta):
      ieta = np.digitize((np.abs(eta),), eta_bins[1:])[0]  # skip lowest edge
      ieta = np.clip(ieta, 0, len(eta_bins)-2)
      return ieta

    nzones = len(eta_bins) - 1
    out = {}  # dict of dict
    for zone in range(nzones):
      out[zone] = {}

    # __________________________________________________________________________
    # Load tree
    tree = load_pgun_batch(jobid)
    verbosity = 1

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      # Particles (pT > 4 GeV)
      part = evt.particles[0]  # particle gun
      if not (part.pt > 4):
        continue

      zone = find_eta_bin(part.eta)

      # Trigger primitives
      for ihit, hit in enumerate(evt.hits):
        lay = find_emtf_layer(hit)
        d = out[zone]
        if lay not in d:
          d[lay] = []
        d[lay].append(hit.emtf_theta)

    # End loop over events

    # __________________________________________________________________________
    # Print results
    for zone in range(nzones):
      d = out[zone]
      keys = sorted(d.keys())
      for k in keys:
        lay = k
        alist = d[lay]
        n = len(alist)
        if n > 100:
          p = np.percentile(alist, [1,2,2.5,3], overwrite_input=True)
          q = np.percentile(alist, [97,97.5,98,99], overwrite_input=True)
          print(zone, '%03i' % lay, '%5i' % n, p, q)
    return

# ______________________________________________________________________________
class UnittestAnalysis(_BaseAnalysis):
  """Unit testing.

  Description.
  """

  def run(self, algo):
    out_part = []
    out_hits = []
    out_tracks = []

    # __________________________________________________________________________
    # Load tree
    def load_pgun_test():
      infile = '../test7/ntuple.root'
      return load_tree(infile)

    tree = load_pgun_test()
    verbosity = 1

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      # Particles
      ievt_part = None
      for ipart, part in enumerate(evt.particles):
        part_id = (part.pdgid,)
        part_val = (part.pt, part.eta, part.phi, part.invpt, part.d0)
        ievt_part = (part_id + part_val)

      # Trigger primitives
      ievt_hits = []
      for ihit, hit in enumerate(evt.hits):
        hit_id = (hit.type, hit.station, hit.ring, find_endsec(hit.endcap, hit.sector), hit.bx)
        hit_val = (hit.emtf_phi, hit.emtf_theta, hit.bend, hit.quality, hit.time, hit.strip, hit.wire, hit.roll, hit.fr, hit.sim_tp1, hit.sim_tp2)
        ievt_hits.append(hit_id + hit_val)

      # Tracks
      ievt_tracks = []
      for itrk, trk in enumerate(evt.tracks):
        trk_id = (find_endsec(trk.endcap, trk.sector), trk.mode)
        trk_val = (trk.pt, trk.xml_pt, trk.phi, trk.theta, trk.eta, trk.q, trk.bx, trk.nhits, trk.hitref1, trk.hitref2, trk.hitref3, trk.hitref4)
        ievt_tracks.append(trk_id + trk_val)

      # Finally, add to output
      out_part.append(ievt_part)
      out_hits.append(ievt_hits)
      out_tracks.append(ievt_tracks)

    # End loop over events

    # __________________________________________________________________________
    # Output
    outfile = 'unittest.npz'
    if use_condor:
      outfile = outfile[:-4] + ('_%i.npz' % jobid)
    out_part = np.asarray(out_part)
    out_hits = create_ragged_array(out_hits)
    out_tracks = create_ragged_array(out_tracks)
    print('[INFO] out_part: {0} out_hits: {1} out_tracks: {2}'.format(out_part.shape, out_hits.shape, out_tracks.shape))
    out_dict = {
      'out_part': out_part,
      'out_hits_values': out_hits.values,
      'out_hits_row_splits': out_hits.row_splits,
      'out_tracks_values': out_tracks.values,
      'out_tracks_row_splits': out_tracks.row_splits,
    }
    save_np_arrays(outfile, out_dict)

    # __________________________________________________________________________
    # Assert
    def load_unittest(f):
      with np.load(f) as loaded:
        out_part = loaded['out_part']
        out_hits = RaggedTensorValue(loaded['out_hits_values'], loaded['out_hits_row_splits'])
        out_tracks = RaggedTensorValue(loaded['out_tracks_values'], loaded['out_tracks_row_splits'])
      return (out_part, out_hits, out_tracks)

    reference = '../test7/unittest_RelVal111X.npz'
    out_part_ref, out_hits_ref, out_tracks_ref = load_unittest(reference)
    print('[INFO] out_part_ref: {0} out_hits_ref: {1} out_tracks_ref: {2}'.format(out_part_ref.shape, out_hits_ref.shape, out_tracks_ref.shape))

    import numpy.testing as npt
    npt.assert_allclose(out_part, out_part_ref)
    print('[INFO] Test 1 OK')
    npt.assert_allclose(out_hits.values, out_hits.values)
    print('[INFO] Test 2 OK')
    npt.assert_allclose(out_tracks.values, out_tracks.values)
    print('[INFO] Test 3 OK')
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
analysis = 'unittest'

# Job id (pick an integer)
jobid = 0

# Max num of events (-1 means all events)
maxevents = -1

# Condor or not
# if 'CONDOR_EXEC' is defined, overwrite the 3 arguments (algo, analysis, jobid)
use_condor = ('CONDOR_EXEC' in os.environ)
if use_condor:
  os.environ['ROOTPY_GRIDMODE'] = 'true'
  algo = sys.argv[1]
  analysis = sys.argv[2]
  jobid = int(sys.argv[3])
  maxevents = -1

# Main function
def main():
  start_time = datetime.datetime.now()
  print('[INFO] Current time    : {0}'.format(start_time))
  print('[INFO] Using cmssw     : {0}'.format(os.environ['CMSSW_VERSION']))
  print('[INFO] Using condor    : {0}'.format(use_condor))
  print('[INFO] Using algo      : {0}'.format(algo))
  print('[INFO] Using analysis  : {0}'.format(analysis))
  print('[INFO] Using jobid     : {0}'.format(jobid))
  print('[INFO] Using maxevents : {0}'.format(maxevents))

  if analysis == 'dummy':
    anna = DummyAnalysis()
    anna.run(algo=algo)

  elif analysis == 'zone':
    anna = ZoneAnalysis()
    anna.run(algo=algo)

  elif analysis == 'unittest':
    anna = UnittestAnalysis()
    anna.run(algo=algo)

  else:
    raise RuntimeError('Cannot recognize analysis: {0}'.format(analysis))

  # DONE!
  stop_time = datetime.datetime.now()
  print('[INFO] Elapsed time    : {0}'.format(stop_time - start_time))

if __name__ == '__main__':
  main()
