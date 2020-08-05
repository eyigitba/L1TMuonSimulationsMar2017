"""Ntuples for EMTF++."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import six
from six.moves import range, zip, map, filter

# ______________________________________________________________________________
# Ntuples

def load_tree(infiles):
  from rootpy.tree import TreeChain
  from rootpy.ROOT import gROOT
  gROOT.SetBatch(True)

  if isinstance(infiles, list) and len(infiles) > 10:
    infiles_trunc = infiles[:5] + ['...'] + infiles[-5:]
  else:
    infiles_trunc = infiles
  print('[INFO] Opening files: {0}'.format(infiles_trunc))
  tree = TreeChain('ntupler/tree', infiles)
  tree.define_collection(name='hits', prefix='vh_', size='vh_size')
  tree.define_collection(name='simhits', prefix='vc_', size='vc_size')
  tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
  tree.define_collection(name='particles', prefix='vp_', size='vp_size')
  return tree

eos_prefix = 'root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_6_3/'

def load_pgun_test():
  #infile = '../test7/ntuple_SingleMuon_Endcap_2GeV_add.6.root'
  infile = '../test7/ntuple_SingleMuon_Displaced_2GeV_PhaseIITDRSpring19_add.8.root'
  return load_tree(infile)

def load_pgun_batch(k):
  my_range = np.split(np.arange(1000), 100)[k]
  infiles = []
  #for i in my_range:
  #  infiles.append(eos_prefix + 'SingleMuon_Endcap_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/190923_203236/%04i/ntuple_SingleMuon_Endcap_%i.root' % ((i+1)//1000, (i+1)))
  #  infiles.append(eos_prefix + 'SingleMuon_Endcap2_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/190923_203346/%04i/ntuple_SingleMuon_Endcap2_%i.root' % ((i+1)//1000, (i+1)))
  for i in my_range:
    infiles.append(eos_prefix + 'SingleMuon_Endcap_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/191115_161447/%04i/ntuple_%i.root' % ((i+1)//1000, (i+1)))
    infiles.append(eos_prefix + 'SingleMuon_Endcap2_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/191115_172000/%04i/ntuple_%i.root' % ((i+1)//1000, (i+1)))
  tree = load_tree(infiles)
  return tree

def load_pgun_displ_batch(k):
  my_range = np.split(np.arange(2000), 200)[k]
  infiles = []
  #for i in my_range:
  #  infiles.append(eos_prefix + 'SingleMuon_Displaced_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/190923_212343/%04i/ntuple_SingleMuon_Displaced_%i.root' % ((i+1)//1000, (i+1)))
  #  infiles.append(eos_prefix + 'SingleMuon_Displaced2_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/190923_212452/%04i/ntuple_SingleMuon_Displaced2_%i.root' % ((i+1)//1000, (i+1)))
  for i in my_range:
    if i < 1000:
      infiles.append(eos_prefix + 'SingleMuon_Displaced_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/191112_182211/%04i/ntuple_%i.root' % ((i+1)//1000, (i+1)))
      infiles.append(eos_prefix + 'SingleMuon_Displaced2_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/191112_224608/%04i/ntuple_%i.root' % ((i+1)//1000, (i+1)))
    else:
      infiles.append(eos_prefix + 'SingleMuon_Displaced_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/191120_015414/%04i/ntuple_%i.root' % ((i+1-1000)//1000, (i+1-1000)))
      infiles.append(eos_prefix + 'SingleMuon_Displaced2_2GeV_PhaseIITDRSpring19/ParticleGuns/CRAB3/191120_020405/%04i/ntuple_%i.root' % ((i+1-1000)//1000, (i+1-1000)))
  tree = load_tree(infiles)
  return tree

def load_mixing_batch(k):
  infiles = []
  #infiles += [eos_prefix + 'ntuple_SingleNeutrino_PU140_PhaseIITDRSpring19/Nu_E10-pythia8-gun/CRAB3/190926_145646/0000/ntuple_%i.root' % (i+1) for i in range(30)]  # up to 30/63
  #infiles += [eos_prefix + 'ntuple_SingleNeutrino_PU200_PhaseIITDRSpring19/Nu_E10-pythia8-gun/CRAB3/190926_145529/0000/ntuple_%i.root' % (i+1) for i in range(40)]  # up to 40/85
  #infiles += [eos_prefix + 'ntuple_SingleNeutrino_PU250_PhaseIITDRSpring19/Nu_E10-pythia8-gun/CRAB3/190926_145757/0000/ntuple_%i.root' % (i+1) for i in range(50)]  # up to 50/125
  #infiles += [eos_prefix + 'ntuple_SingleNeutrino_PU300_PhaseIITDRSpring19/Nu_E10-pythia8-gun/CRAB3/191002_214457/0000/ntuple_%i.root' % (i+1) for i in range(50)]  # up to 50/111
  infiles += [eos_prefix + 'ntuple_DoubleElectron_PU140_PhaseIITDRSpring19/DoubleElectron_FlatPt-1To100/CRAB3/190925_044352/0000/ntuple_%i.root' % (i+1) for i in range(25)]
  infiles += [eos_prefix + 'ntuple_DoubleElectron_PU200_PhaseIITDRSpring19/DoubleElectron_FlatPt-1To100/CRAB3/190925_044710/0000/ntuple_%i.root' % (i+1) for i in range(25)]
  infiles += [eos_prefix + 'ntuple_DoublePhoton_PU140_PhaseIITDRSpring19/DoublePhoton_FlatPt-1To100/CRAB3/190925_044839/0000/ntuple_%i.root' % (i+1) for i in range(25)]
  infiles += [eos_prefix + 'ntuple_DoublePhoton_PU200_PhaseIITDRSpring19/DoublePhoton_FlatPt-1To100/CRAB3/190925_044947/0000/ntuple_%i.root' % (i+1) for i in range(25)]
  infiles += [eos_prefix + 'ntuple_SingleElectron_PU200_PhaseIITDRSpring19/SingleElectron_PT2to100/CRAB3/190925_181236/0000/ntuple_%i.root' % (i+1) for i in range(15)]
  infiles += [eos_prefix + 'ntuple_SinglePhoton_PU200_PhaseIITDRSpring19/PhotonFlatPt8To150/CRAB3/190925_181357/0000/ntuple_%i.root' % (i+1) for i in range(77)]
  infile = infiles[k]
  tree = load_tree(infile)
  return tree

def load_singleneutrino_batch(k, pileup=200):
  if pileup == 140:
    infiles = [eos_prefix + 'ntuple_SingleNeutrino_PU140_PhaseIITDRSpring19/Nu_E10-pythia8-gun/CRAB3/190926_145646/0000/ntuple_%i.root' % (i+1) for i in range(63)]
  elif pileup == 200:
    infiles = [eos_prefix + 'ntuple_SingleNeutrino_PU200_PhaseIITDRSpring19/Nu_E10-pythia8-gun/CRAB3/190926_145529/0000/ntuple_%i.root' % (i+1) for i in range(85)]
  elif pileup == 250:
    infiles = [eos_prefix + 'ntuple_SingleNeutrino_PU250_PhaseIITDRSpring19/Nu_E10-pythia8-gun/CRAB3/190926_145757/0000/ntuple_%i.root' % (i+1) for i in range(125)]
  elif pileup == 300:
    infiles = [eos_prefix + 'ntuple_SingleNeutrino_PU300_PhaseIITDRSpring19/Nu_E10-pythia8-gun/CRAB3/191002_214457/0000/ntuple_%i.root' % (i+1) for i in range(111)]
  else:
    raise RuntimeError('Cannot recognize pileup: {0}'.format(pileup))
  infile = infiles[k]
  tree = load_tree(infile)
  return tree

def load_mumu_flatpt_batch(k, pileup=200):
  if pileup == 0:
    infiles = [eos_prefix + 'ntuple_MuMu_FlatPt_PU0_PhaseIITDRSpring19/Mu_FlatPt2to100-pythia8-gun/CRAB3/190925_042003/0000/ntuple_MuMu_FlatPt_PU0_%i.root' % (i+1) for i in range(5)]
  elif pileup == 200:
    infiles = [eos_prefix + 'ntuple_MuMu_FlatPt_PU200_PhaseIITDRSpring19/Mu_FlatPt2to100-pythia8-gun/CRAB3/190925_051735/0000/ntuple_%i.root' % (i+1) for i in range(33)]
  elif pileup == 300:
    infiles = [eos_prefix + 'ntuple_MuMu_FlatPt_PU300_PhaseIITDRSpring19/Mu_FlatPt2to100-pythia8-gun/CRAB3/190924_201214/0000/ntuple_MuMu_FlatPt_PU300_%i.root' % (i+1) for i in range(280)]
  else:
    raise RuntimeError('Cannot recognize pileup: {0}'.format(pileup))
  infile = infiles[k]
  tree = load_tree(infile)
  return tree
