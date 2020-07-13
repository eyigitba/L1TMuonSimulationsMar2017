"""Data analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from emtf_algos import *
from emtf_ntuples import *


# ______________________________________________________________________________
# Analyses

class _BaseAnalysis(object):
  """Abstract base class"""
  pass

# ______________________________________________________________________________
class MuonPerfAnalysis(_BaseAnalysis):
  """Make performance plots using muon dataset.

  Description.
  """

  def run(self, algo, pileup=200):
    return

# ______________________________________________________________________________
class NeutrinoPerfAnalysis(_BaseAnalysis):
  """Make performance plots using neutrino dataset.

  Description.
  """

  def run(self, algo, pileup=200):
    return


# ______________________________________________________________________________
# Main

import os, sys, datetime

# Algorithm (pick one)
algo = 'default'  # phase 2
#algo = 'run3'

# Analysis mode (pick one)
analysis = 'muon'
#analysis = 'neutrino'

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
  if analysis == 'muon':
    myapp = MuonPerfAnalysis()
    myargs = dict(algo=algo)
  elif analysis == 'muon0':
    myapp = MuonPerfAnalysis()
    myargs = dict(algo=algo, pileup=0)
  elif analysis == 'muon200':
    myapp = MuonPerfAnalysis()
    myargs = dict(algo=algo, pileup=200)
  elif analysis == 'muon300':
    myapp = MuonPerfAnalysis()
    myargs = dict(algo=algo, pileup=300)

  elif analysis == 'neutrino':
    myapp = NeutrinoPerfAnalysis()
    myargs = dict(algo=algo)
  elif analysis == 'neutrino140':
    myapp = NeutrinoPerfAnalysis()
    myargs = dict(algo=algo, pileup=140)
  elif analysis == 'neutrino200':
    myapp = NeutrinoPerfAnalysis()
    myargs = dict(algo=algo, pileup=200)
  elif analysis == 'neutrino250':
    myapp = NeutrinoPerfAnalysis()
    myargs = dict(algo=algo, pileup=250)
  elif analysis == 'neutrino300':
    myapp = NeutrinoPerfAnalysis()
    myargs = dict(algo=algo, pileup=300)

  else:
    raise RuntimeError('Cannot recognize analysis: {0}'.format(analysis))

  # Run analysis
  myapp.run(**myargs)
  return

# Finally
if __name__ == '__main__':
  app()
