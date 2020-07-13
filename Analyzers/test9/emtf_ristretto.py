"""Data preparation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from emtf_algos import *
from emtf_ntuples import *


# ______________________________________________________________________________
# Classes

class EMTFSectorRanking(object):
  def __init__(self):
    self.sectors = np.zeros(num_emtf_sectors, dtype=np.int32)

  def reset(self):
    self.sectors.fill(0)

  def add(self, hit):
    ri_layer = find_emtf_ri_layer(hit.type, hit.station, hit.ring)
    assert (0 <= ri_layer and ri_layer <= 18)
    ri_layer_valid = np.zeros(8, dtype=np.bool)
    ri_layer_valid[0] = (ri_layer == 18)              # ME0
    ri_layer_valid[1] = (ri_layer == 0)               # ME1/1
    ri_layer_valid[2] = (ri_layer in (1,2))           # ME1/2, ME1/3
    ri_layer_valid[3] = (ri_layer in (3,4))           # ME2/1, ME2/2
    ri_layer_valid[4] = (ri_layer in (5,6))           # ME3/1, ME3/2
    ri_layer_valid[5] = (ri_layer in (7,8))           # ME4/1, ME4/2
    ri_layer_valid[6] = (ri_layer in (9,10,11,12,13)) # GE1/1, RE1/2, GE2/1, RE2/2
    ri_layer_valid[7] = (ri_layer in (14,15,16,17))   # RE3/1, RE3/2, RE4/1, RE4/2
    rank = np.packbits(ri_layer_valid)

    endsec = get_trigger_endsec(hit.endcap, hit.sector)
    self.sectors[endsec] |= rank

  def get_best_sector(self):
    best_sector = np.argmax(self.sectors)
    best_sector_rank = self.sectors[best_sector]
    return (best_sector, best_sector_rank)

class EMTFChamberCouncil(object):
  def __init__(self, is_sim=False):
    self.chambers = {}
    self.is_sim = is_sim

  def reset(self):
    self.chambers.clear()

  def add(self, hit):
    # Call functions
    emtf_layer = find_emtf_layer(hit.type, hit.station, hit.ring)
    ri_layer = find_emtf_ri_layer(hit.type, hit.station, hit.ring)
    emtf_chamber = find_emtf_chamber(hit.type, hit.station, hit.cscid, hit.subsector, hit.neighbor)
    emtf_segment = 0
    zones = find_emtf_zones(ri_layer, hit.emtf_theta)
    timezones = find_emtf_timezones(ri_layer, hit.bx)
    try:
      detlayer = hit.layer
    except:
      detlayer = 0

    emtf_phi = find_emtf_phi(hit)
    emtf_theta = find_emtf_theta(hit)
    emtf_theta_alt = emtf_theta
    emtf_bend = find_emtf_bend(hit)
    emtf_qual = find_emtf_qual(hit)
    emtf_time = find_emtf_time(hit)

    # Assign variables
    hit.emtf_layer = emtf_layer
    hit.ri_layer = ri_layer
    hit.emtf_chamber = emtf_chamber
    hit.emtf_segment = emtf_segment
    hit.zones = zones
    hit.timezones = timezones
    hit.detlayer = detlayer

    hit.emtf_phi = emtf_phi
    hit.emtf_theta = emtf_theta
    hit.emtf_theta_alt = emtf_theta_alt
    hit.emtf_bend = emtf_bend
    hit.emtf_qual = emtf_qual
    hit.emtf_time = emtf_time

    # Add hit to chamber
    k = (hit.bx, hit.emtf_chamber)
    if k not in self.chambers:
      self.chambers[k] = []
    self.chambers[k].append(hit)

  def _to_numpy(self, hits):
    default_value = -99
    getter = lambda hit: [hit.emtf_layer, hit.ri_layer, hit.zones, hit.timezones,
                          hit.emtf_chamber, hit.emtf_segment, hit.detlayer, hit.bx,
                          hit.emtf_phi, hit.emtf_bend, hit.emtf_theta, hit.emtf_theta_alt,
                          hit.emtf_qual, hit.emtf_time, hit.fr, default_value]
    arr = np.array([getter(hit) for hit in hits], dtype=np.int32)
    return arr

  def _get_hits_from_chambers_sim(self):
    hits = []
    sorted_keys = sorted(self.chambers.keys())
    for k in sorted_keys:
      (bx, emtf_chamber) = k
      tmp_hits = self.chambers[k]

      # If more than one hit, sort by layer number.
      # For RPC and GEM, keep the first one; For CSC, ME0 and DT, keep the median
      ind = 0
      if len(tmp_hits) > 1:
        tmp_hits.sort(key=lambda hit: hit.layer)
        if tmp_hits[0].type == kRPC or tmp_hits[0].type == kGEM:
          ind = 0
        else:
          ind = (len(tmp_hits)-1)//2

      hit = tmp_hits[ind]
      hits.append(hit)
    return hits

  def _get_hits_from_chambers(self):
    hits = []
    sorted_keys = sorted(self.chambers.keys())
    for k in sorted_keys:
      (bx, emtf_chamber) = k
      tmp_hits = self.chambers[k]

      if emtf_chamber < 54:  # CSC
        assert len(tmp_hits) in (1,2,4)
        emtf_phi_a = np.min([hit.emtf_phi for hit in tmp_hits])
        emtf_phi_b = np.max([hit.emtf_phi for hit in tmp_hits])
        emtf_theta_a = np.min([hit.emtf_theta for hit in tmp_hits])
        emtf_theta_b = np.max([hit.emtf_theta for hit in tmp_hits])

        emtf_segment = 0
        for hit in tmp_hits:
          keep = False
          if (hit.emtf_phi == emtf_phi_a) and (hit.emtf_theta == emtf_theta_a):
            keep = True
            hit.emtf_theta_alt = emtf_theta_b
          elif (hit.emtf_phi == emtf_phi_b) and (hit.emtf_theta == emtf_theta_b):
            keep = True
            hit.emtf_theta_alt = emtf_theta_a

          if keep:
            hit.emtf_segment = emtf_segment
            emtf_segment += 1
            hits.append(hit)

      else:  # non-CSC
        emtf_segment = 0
        for hit in tmp_hits:
          hit.emtf_segment = emtf_segment
          emtf_segment += 1
          hits.append(hit)
    return hits

  def get_hits(self):
    if self.is_sim:
      hits = self._get_hits_from_chambers_sim()
    else:
      hits = self._get_hits_from_chambers()
    hits = self._to_numpy(hits)
    return hits


# ______________________________________________________________________________
# Analyses

class _BaseAnalysis(object):
  """Abstract base class"""
  pass

# ______________________________________________________________________________
class SignalAnalysis(_BaseAnalysis):
  """Prepare signal data used for training.

  Description.
  """

  def run(self, algo, signal='prompt'):
    out_part = []
    out_hits = []
    out_simhits = []

    sectors = EMTFSectorRanking()
    chambers = EMTFChamberCouncil()
    chambers_simhits = EMTFChamberCouncil(is_sim=True)

    # __________________________________________________________________________
    # Load tree
    if signal == 'prompt':
      tree = load_pgun_batch(jobid)
    elif signal == 'displ':
      tree = load_pgun_displ_batch(jobid)
    else:
      raise RuntimeError('Unexpected signal: {0}'.format(signal))

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      if (ievt % 1000) == 0:
        print('Processing event {0}'.format(ievt))

      # Reset
      sectors.reset()
      chambers.reset()
      chambers_simhits.reset()

      # Particles
      try:
        part = evt.particles[0]  # particle gun
      except:
        continue

      # First, determine the best sector

      # Trigger primitives
      for ihit, hit in enumerate(evt.hits):
        if is_emtf_legit_hit(hit):
          sectors.add(hit)

      (best_sector, best_sector_rank) = sectors.get_best_sector()

      # Second, fill the chambers with trigger primitives

      # Trigger primitives
      for ihit, hit in enumerate(evt.hits):
        if is_emtf_legit_hit(hit) and get_trigger_endsec(hit.endcap, hit.sector) == best_sector:
          if min_emtf_strip <= hit.emtf_phi < max_emtf_strip:
            chambers.add(hit)

      # Third, fill the chambers with sim hits

      # Sim hits
      for isimhit, simhit in enumerate(evt.simhits):
        if is_emtf_legit_hit(simhit):
          if simhit.type == kME0:
            # Special case for ME0 as it is a 20-deg chamber. Pretend it is ME2/1 when calling these functions.
            fake_station, fake_ring = 2, 1
            simhit.endcap = +1 if simhit.z >= 0 else -1
            simhit.sector = get_trigger_sector(fake_ring, fake_station, simhit.chamber)
            simhit.subsector = get_trigger_subsector(fake_ring, fake_station, simhit.chamber)
            simhit.cscid = get_trigger_cscid(fake_ring, fake_station, simhit.chamber)
            simhit.neighid = get_trigger_neighid(fake_ring, fake_station, simhit.chamber)
          else:
            simhit.endcap = +1 if simhit.z >= 0 else -1
            simhit.sector = get_trigger_sector(simhit.ring, simhit.station, simhit.chamber)
            simhit.subsector = get_trigger_subsector(simhit.ring, simhit.station, simhit.chamber)
            simhit.cscid = get_trigger_cscid(simhit.ring, simhit.station, simhit.chamber)
            simhit.neighid = get_trigger_neighid(simhit.ring, simhit.station, simhit.chamber)

          simhit.emtf_phi = calc_phi_loc_int(np.rad2deg(simhit.phi), (best_sector%6) + 1)
          simhit.emtf_theta = calc_theta_int(np.rad2deg(simhit.theta), 1 if best_sector < 6 else -1)

          simhit.bx, simhit.bend, simhit.quality, simhit.fr, simhit.time = 0, 0, 0, 0, 0  # dummy
          simhit.neighbor = 0

          # Also need to send to the next sector
          get_prev_sector = lambda sector: sector - 1 if sector != 1 else 6
          get_next_sector = lambda sector: sector + 1 if sector != 6 else 1
          if get_trigger_endsec(simhit.endcap, simhit.sector) == best_sector:
            chambers_simhits.add(simhit)
          elif get_trigger_endsec(simhit.endcap, get_next_sector(simhit.sector)) == best_sector:
            simhit.neighbor = 1
            chambers_simhits.add(simhit)
          elif get_trigger_endsec(simhit.endcap, get_prev_sector(simhit.sector)) == best_sector:
            if simhit.type == kME0 and simhit.emtf_phi >= ((55 + 22) * 60):
              # Special case for ME0 as there is a 5 deg shift.
              # The CSC chamber 1 starts at -5 deg, but the ME0 chamber 1 starts at -10 deg.
              chambers_simhits.add(simhit)

      # Fourth, extract the particle and hits
      def get_part_info():
        etastar = calc_etastar_from_eta(part.invpt, part.eta, part.phi, part.vx, part.vy, part.vz)
        part_zone = find_particle_zone(etastar)
        part_info = [part.invpt, part.eta, part.phi, part.vx, part.vy, part.vz, part.d0, best_sector, part_zone]
        part_info = np.array(part_info, dtype=np.float32)
        return part_info

      ievt_part = get_part_info()
      ievt_hits = chambers.get_hits()
      ievt_simhits = chambers_simhits.get_hits()

      # Finally, add to output
      out_part.append(ievt_part)
      out_hits.append(ievt_hits)
      out_simhits.append(ievt_simhits)

      # Debug
      if verbosity >= kINFO:
        print('Processing event {0}'.format(ievt))
        print('.. part {0} {1} {2} {3} {4} {5}'.format(0, part.pt, part.eta, part.phi, part.invpt, part.d0))
        for ihit, hit in enumerate(evt.hits):
          hit_id = (hit.type, hit.station, hit.ring, get_trigger_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
          hit_sim_tp = hit.sim_tp1
          if (hit.type == kCSC) and (hit_sim_tp != hit.sim_tp2):
            hit_sim_tp = -1
          print('.. hit {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(ihit, hit_id, hit.emtf_phi, hit.emtf_theta, hit.bend, hit.quality, hit_sim_tp, hit.strip, hit.wire))
        for isimhit, simhit in enumerate(evt.simhits):
          simhit_id = (simhit.type, simhit.station, simhit.ring, get_trigger_endsec(simhit.endcap, simhit.sector), simhit.fr, simhit.bx)
          print('.. simhit {0} {1} {2} {3}'.format(isimhit, simhit_id, simhit.phi, simhit.theta))
        print('best sector: {0} rank: {1}'.format(best_sector, best_sector_rank))
        with np.printoptions(linewidth=100, threshold=1000):
          print('hits:')
          print(ievt_hits)
          print('simhits:')
          print(ievt_simhits)

    # End loop over events

    # __________________________________________________________________________
    # Output
    outfile = 'signal.npz'
    if use_condor:
      outfile = outfile[:-4] + ('_%i.npz' % jobid)
    out_part = np.asarray(out_part)
    out_hits = create_ragged_array(out_hits)
    out_simhits = create_ragged_array(out_simhits)
    print('[INFO] out_part: {0} out_hits: {1} out_simhits: {2}'.format(out_part.shape, out_hits.shape, out_simhits.shape))
    out_dict = {
      'out_part': out_part,
      'out_hits_values': out_hits.values,
      'out_hits_row_splits': out_hits.row_splits,
      'out_simhits_values': out_simhits.values,
      'out_simhits_row_splits': out_simhits.row_splits,
    }
    save_np_arrays(outfile, out_dict)
    return

# ______________________________________________________________________________
class BkgndAnalysis(_BaseAnalysis):
  """Prepare background data used for training.

  Description.
  """

  def run(self, algo):
    out_aux = []
    out_hits = []

    sector_chambers = [EMTFChamberCouncil() for sector in range(num_emtf_sectors)]

    # __________________________________________________________________________
    # Load tree
    tree = load_mixing_batch(jobid)

    # Loop over events
    for ievt, evt in enumerate(tree):
      if maxevents != -1 and ievt == maxevents:
        break

      if (ievt % 100) == 0:
        print('Processing event {0}'.format(ievt))

      # Reset
      for sector in range(num_emtf_sectors):
        sector_chambers[sector].reset()

      # First, apply event veto

      # Particles
      veto = False
      for ipart, part in enumerate(evt.particles):
        if (part.bx == 0) and (part.pt > 14.) and (find_particle_zone(part.eta) in (0,1,2,3)):
          veto = True
          break
      if veto:
        continue

      # Second, fill the zones with trigger primitives

      # Trigger primitives
      for sector in range(num_emtf_sectors):
        for ihit, hit in enumerate(evt.hits):
          if is_emtf_legit_hit(hit) and get_trigger_endsec(hit.endcap, hit.sector) == sector:
            if min_emtf_strip <= hit.emtf_phi < max_emtf_strip:
              sector_chambers[sector].add(hit)

      # Finally, extract the particle and hits. Add them to output.
      def get_aux_info():
        aux_info = [jobid, ievt, sector]
        aux_info = np.array(aux_info, dtype=np.int32)
        return aux_info

      for sector in range(num_emtf_sectors):
        ievt_aux = get_aux_info()
        ievt_hits = sector_chambers[sector].get_hits()
        out_aux.append(ievt_aux)
        out_hits.append(ievt_hits)

      # Debug
      if verbosity >= kINFO:
        print('Processing event {0}'.format(ievt))
        for ihit, hit in enumerate(evt.hits):
          hit_id = (hit.type, hit.station, hit.ring, get_trigger_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
          hit_sim_tp = hit.sim_tp1
          if (hit.type == kCSC) and (hit_sim_tp != hit.sim_tp2):
            hit_sim_tp = -1
          print('.. hit {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(ihit, hit_id, hit.emtf_phi, hit.emtf_theta, hit.bend, hit.quality, hit_sim_tp, hit.strip, hit.wire))
        with np.printoptions(linewidth=100, threshold=1000):
          for sector in range(num_emtf_sectors):
            ievt_hits = sector_chambers[sector].get_hits()
            print('sector {0} hits:'.format(sector))
            print(ievt_hits)

    # End loop over events

    # __________________________________________________________________________
    # Output
    outfile = 'bkgnd.npz'
    if use_condor:
      outfile = outfile[:-4] + ('_%i.npz' % jobid)
    out_aux = np.asarray(out_aux)
    out_hits = create_ragged_array(out_hits)
    print('[INFO] out_hits: {0}'.format(out_hits.shape))
    out_dict = {
      'out_aux': out_aux,
      'out_hits_values': out_hits.values,
      'out_hits_row_splits': out_hits.row_splits,
    }
    save_np_arrays(outfile, out_dict)
    return


# ______________________________________________________________________________
# Main

import os, sys, datetime

# Algorithm (pick one)
algo = 'default'  # phase 2
#algo = 'run3'

# Analysis mode (pick one)
analysis = 'signal'
#analysis = 'signal_displ'
#analysis = 'bkgnd'

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
  if analysis == 'signal':
    myapp = SignalAnalysis()
    myargs = dict(algo=algo, signal='prompt')
  elif analysis == 'signal_displ':
    myapp = SignalAnalysis()
    myargs = dict(algo=algo, signal='displ')

  elif analysis == 'bkgnd':
    myapp = BkgndAnalysis()
    myargs = dict(algo=algo)

  else:
    raise RuntimeError('Cannot recognize analysis: {0}'.format(analysis))

  # Run analysis
  myapp.run(**myargs)
  return

# Finally
if __name__ == '__main__':
  app()
