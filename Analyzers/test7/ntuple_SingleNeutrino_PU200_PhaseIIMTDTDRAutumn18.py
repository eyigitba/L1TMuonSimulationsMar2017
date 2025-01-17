# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 --step L1 --mc --eventcontent FEVTDEBUGHLT --datatier GEN-SIM-DIGI-RAW --conditions auto:phase2_realistic --geometry Extended2023D35 --era Phase2C4_timing_layer_bar --filein file:step0.root --fileout file:step1.root --no_exec --nThreads 4 -n 100
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C4_timing_layer_bar_cff import Phase2C4_timing_layer_bar

process = cms.Process('L1',Phase2C4_timing_layer_bar)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D35Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/PhaseIIMTDTDRAutumn18DR/NeutrinoGun_E_10GeV/FEVT/PU200_103X_upgrade2023_realistic_v2-v1/40000/DA20A045-9075-4240-BC0E-FBFAB6F65484.root',
    ),
    secondaryFileNames = cms.untracked.vstring(),
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('l1NtupleMC nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('l1NtupleMC_L1.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Path and EndPath definitions
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

## Schedule definition
#process.schedule = cms.Schedule(process.L1simulation_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
#from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
#associatePatAlgosToolsTask(process)

##Setup FWK for multithreaded
#process.options.numberOfThreads=cms.untracked.uint32(4)
#process.options.numberOfStreams=cms.untracked.uint32(0)
#process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion


# ______________________________________________________________________________
# Modify input files
if True:
  from L1TMuonSimulations.Configuration.tools import *
  txt = 'L1TMuonSimulations/Configuration/data/input_NeutrinoGun_E_10GeV_PhaseIIMTDTDRAutumn18_PU200.txt'
  txt = os.path.join(os.environ['CMSSW_BASE'], 'src', txt)
  fileNames_txt = loadFromFile(txt, fmt='')
  process.source.fileNames = fileNames_txt

# ______________________________________________________________________________
# Modify EMTF
if True:
    from L1Trigger.L1TMuonEndCap.customise_Phase2 import customise as customise_Phase2
    process = customise_Phase2(process)

# ______________________________________________________________________________
# Modify paths and schedule definitions
print("[INFO] Using GlobalTag: %s" % process.GlobalTag.globaltag.value())
if True:
    # Ntuplize
    process.load('L1TMuonSimulations.Analyzers.ntupler_cfi')
    process.TFileService = cms.Service('TFileService', fileName = process.ntupler.outFileName)
    # Modify sequences without any consequences
    #process.doAllDigiTask = cms.Task(process.generatorSmeared, process.muonDigiTask)
    process.SimL1TMuonTask = cms.Task(process.SimL1TMuonCommonTask, process.me0TriggerPseudoDigiTask, process.me0TriggerPseudoDigiTask105X, process.rpcRecHits, process.simBmtfDigis, process.simEmtfDigis, process.simOmtfDigis, process.simTwinMuxDigis)
    process.SimL1EmulatorCoreTask = cms.Task(process.SimL1TMuonTask)
    process.SimL1EmulatorTask = cms.Task(process.SimL1EmulatorCoreTask)
    process.ntuple_step = cms.Path(process.ntupler)
    process.schedule = cms.Schedule(process.L1simulation_step, process.ntuple_step)


# ______________________________________________________________________________
# Configure framework report and summary
process.options.wantSummary = cms.untracked.bool(True)
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# Dump the full python config
with open('dump.py', 'w') as f:
    f.write(process.dumpPython())
