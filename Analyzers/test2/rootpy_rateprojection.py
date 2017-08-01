import numpy as np
np.random.seed(2023)

from rootpy.plotting import Hist, Hist2D, Graph, Efficiency, Legend, Canvas
from rootpy.tree import Tree, TreeModel, TreeChain, FloatCol, IntCol, ShortCol
from rootpy.io import root_open


# ______________________________________________________________________________
# Tree models
#   see: http://www.rootpy.org/auto_examples/tree/model.html

class Hit(TreeModel):
  pass

class Track(TreeModel):
  pass

class Particle(TreeModel):
  pass


# ______________________________________________________________________________
# Analyzer

mystate = 0

# Open file
if mystate == 0:
  tree_name_i = '/home/jlow/L1MuonTrigger/CRAB3/P2_9_2_3_patch1/crab_projects/crab_ntuple_SingleMuon_PositiveEndCap/results/ntuple_SingleMuon_PositiveEndCap_%i.root'
  tree = TreeChain('ntupler/tree', [(tree_name_i % i) for i in range(1,8+1)])
elif mystate == 1:
  tree_name_i = '/home/jlow/L1MuonTrigger/CRAB3/P2_9_2_3_patch1/crab_projects/crab_ntuple_SingleNeutrino_PU140/results/ntuple_SingleNeutrino_PU140_%i.root'
  tree = TreeChain('ntupler/tree', [(tree_name_i % i) for i in range(1,90+1)])
else:
  raise Exception("Unexpected state: %i" % mystate)
#maxEvents = -1
maxEvents = 20000

# Define collection
tree.define_collection(name='hits', prefix='vh_', size='vh_size')
tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
tree.define_collection(name='genparticles', prefix='vp_', size='vp_size')

# Enums
kDT, kCSC, kRPC, kGEM, kTT = 0, 1, 2, 3, 20

# Lambdas
deg_to_rad = lambda x: x * np.pi/180.

rad_to_deg = lambda x: x * 180./np.pi

# Functions
def delta_phi(lhs, rhs):  # in degrees
  deg = lhs - rhs
  while deg <  -180.:  deg += 360.
  while deg >= +180.:  deg -= 360.
  return deg

# Constants
pt_bins = [2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 22., 24., 26., 28., 30., 32., 36., 40., 45., 50., 55., 60., 70., 80., 100., 120., 140., 160., 180., 200., 250., 300., 400., 600., 1000.]

eta_bins = [0., 0.83, 1.24, 1.64, 2.14, 2.5]

# Classes
class EfficiencyMatrix:
  def __init__(self, nbinsx=24, xmin=0., xmax=2.4, xbins=None, nbinsy=100, ymin=0., ymax=100., ybins=None, nbinsz=100, zmin=0., zmax=100., zbins=None):
    """
    x: gen_eta
    y: gen_pt
    z: l1t_pt
    """
    if xbins:  nbinsx = len(xbins)
    if ybins:  nbinsy = len(ybins)
    if zbins:  nbinsz = len(zbins)
    self._nbinsx = nbinsx
    self._xmin = xmin
    self._xmax = xmax
    self._xwidth = (xmax - xmin) / float(nbinsx)
    self._xbins = xbins
    self._nbinsy = nbinsy
    self._ymin = ymin
    self._ymax = ymax
    self._ywidth = (ymax - ymin) / float(nbinsy)
    self._ybins = ybins
    self._nbinsz = nbinsz
    self._zmin = zmin
    self._zmax = zmax
    self._zwidth = (zmax - zmin) / float(nbinsz)
    self._zbins = zbins
    self._numer = np.zeros((nbinsx, nbinsy, nbinsz), dtype=int)
    self._denom = np.zeros((nbinsx, nbinsy, nbinsz), dtype=int)
    self._effie = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)

    # In addition, profile phi and eta
    self._phi_cnt = np.zeros((nbinsx, nbinsy), dtype=int)
    self._phi_mean = np.zeros((nbinsx, nbinsy), dtype=float)
    self._phi_var = np.zeros((nbinsx, nbinsy), dtype=float)
    self._eta_cnt = np.zeros((nbinsx, nbinsy), dtype=int)
    self._eta_mean = np.zeros((nbinsx, nbinsy), dtype=float)
    self._eta_var = np.zeros((nbinsx, nbinsy), dtype=float)

  def sanity_check(self):
    def is_sorted(a):
      if not len(a):
        return True
      else:
        return np.all(np.less_equal(a[:-1], a[1:]))

    if self._xbins:
      assert(is_sorted(self._xbins))
    if self._ybins:
      assert(is_sorted(self._ybins))
    if self._zbins:
      assert(is_sorted(self._zbins))

  def find_binx(self, x):
    if self._xbins:
      binx = np.searchsorted(self._xbins, x)
    else:
      binx = int(np.floor((x - self._xmin) / self._xwidth))
    if binx < 0: binx = 0
    if binx >= self._nbinsx: binx = self._nbinsx - 1
    return binx

  def find_biny(self, y):
    if self._ybins:
      biny = np.searchsorted(self._ybins, y)
    else:
      biny = int(np.floor((y - self._ymin) / self._ywidth))
    if biny < 0: biny = 0
    if biny >= self._nbinsy: biny = self._nbinsy - 1
    return biny

  def find_binz(self, z):
    if self._zbins:
      binz = np.searchsorted(self._zbins, z)
    else:
      binz = int(np.floor((z - self._zmin) / self._zwidth))
    if binz < 0: binz = 0
    if binz >= self._nbinsz: binz = self._nbinsz - 1
    return binz

  def find_edgex(self, binx):
    if self._xbins:
      edgex = self._xbins[binx]
    else:
      edgex = self._xmin + binx * self._xwidth
    return edgex

  def find_edgey(self, biny):
    if self._ybins:
      edgey = self._ybins[biny]
    else:
      edgey = self._ymin + biny * self._ywidth
    return edgey

  def find_edgez(self, binz):
    if self._zbins:
      edgez = self._zbins[binz]
    else:
      edgez = self._zmin + binz * self._zwidth
    return edgez

  def fill(self, gen_eta, gen_pt, l1t_pt):
    binx = self.find_binx(gen_eta)
    biny = self.find_biny(gen_pt)
    binz = self.find_binz(l1t_pt)
    #
    passed = [z >= binz for z in xrange(self._nbinsz)]
    self._denom[binx, biny, :] += 1
    self._numer[binx, biny, passed] += 1

  def profile(self, gen_eta, gen_pt, phi, eta):
    binx = self.find_binx(gen_eta)
    biny = self.find_biny(gen_pt)
    #
    self._phi_cnt[binx, biny] += 1
    delta = (phi - self._phi_mean[binx, biny])
    self._phi_mean[binx, biny] += delta / float(self._phi_cnt[binx, biny])
    self._phi_var[binx, biny] += delta * (phi - self._phi_mean[binx, biny])
    #
    self._eta_cnt[binx, biny] += 1
    delta = (eta - self._eta_mean[binx, biny])
    self._eta_mean[binx, biny] += delta / float(self._eta_cnt[binx, biny])
    self._eta_var[binx, biny] += delta * (eta - self._eta_mean[binx, biny])

  def freeze(self):
    tmp_numer = self._numer.astype(float)
    tmp_denom = self._denom.astype(float)
    tmp_denom = np.where(tmp_denom < 1e-9, 1e-9, tmp_denom)  # avoid division by zero
    np.true_divide(tmp_numer, tmp_denom, out=self._effie)
    #a = self._effie.reshape(-1)
    #for i, v in enumerate(a):
    #  if np.isnan(v):
    #    a_[i] = 0.

  def sitrep(self):
    print self._effie


# Book histograms
histograms = {}
histogram2Ds = {}

em = EfficiencyMatrix(xbins=eta_bins, ybins=pt_bins)
em.sanity_check()


# ______________________________________________________________________________
# Loop over events
for ievt, evt in enumerate(tree):
  if maxEvents != -1 and ievt == maxEvents:
    break

  # ____________________________________________________________________________
  # Verbose

  verbose = False

  if verbose:
    if (ievt % 1 == 0):  print("Processing event: {0}".format(ievt))

    # Hits
    for ihit, hit in enumerate(evt.hits):
      print(".. hit  {0} {1} {2} {3} {4} {5} {6} {7}".format(ihit, hit.bx, hit.type, hit.station, hit.ring, hit.sim_phi, hit.sim_theta, hit.fr))
    # Tracks
    for itrk, trk in enumerate(evt.tracks):
      print(".. trk  {0} {1} {2} {3} {4} {5} {6}".format(itrk, trk.pt, trk.phi, trk.eta, trk.theta, trk.q, trk.mode))
    # Gen particles
    for ipart, part in enumerate(evt.genparticles):
      print(".. part {0} {1} {2} {3} {4} {5}".format(ipart, part.pt, part.phi, part.eta, part.theta, part.q))
  else:
    if (ievt % 1000 == 0):  print("Processing event: {0}".format(ievt))

  # ____________________________________________________________________________
  # Fill efficiency matrix
  if mystate == 0:
    assert len(evt.genparticles) == 1

    mypart = evt.genparticles[0]
    #mytrk = evt.tracks[0]
    try:
      # Select highest pT track from tracks that have a station 1 hit
      mytrk = max(filter(lambda x: (x.mode in [11,13,14,15]), evt.tracks), key=lambda x: x.pt)
    except ValueError, e:
      mytrk = None

    # Fill efficiency matrix
    gen_eta = mypart.eta
    gen_pt = mypart.pt
    if mytrk:
      l1t_pt = mytrk.pt
      l1t_phi = delta_phi(mytrk.phi, rad_to_deg(mypart.phi)) if mytrk.q < 0 else -delta_phi(mytrk.phi, rad_to_deg(mypart.phi))
      l1t_eta = mytrk.eta
    else:
      l1t_pt = 0.
      l1t_phi = 0.
      l1t_eta = 0.

    em.fill(abs(gen_eta), gen_pt, l1t_pt)
    if mytrk:
      em.profile(abs(gen_eta), gen_pt, l1t_phi, abs(l1t_eta))


  # ____________________________________________________________________________
  # Fill efficiency matrix
  elif mystate == 1:

    for ipart, part in enumerate(evt.genparticles):
      pass


  continue  # end loop over event

# ____________________________________________________________________________
# Print efficiency matrix

if mystate == 0:
  em.freeze()
  em.sitrep()
  #np.savetxt('test.out', em._effie, delimiter=',')  # only works for 2D array
  np.savetxt('test.out', em._effie.reshape(-1,em._effie.shape[-1]), delimiter=',')  # reshape to 2D array