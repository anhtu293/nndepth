from nndepth.disparity.models import CREStereo, IGEVStereoMBNet, BaseRAFTStereo, HPRAFTStereo, GroupHPRAFTStereo


MODELS = {
    "crestereo-base": CREStereo,
    "igev-mbnet": IGEVStereoMBNet,
    "raft-base": BaseRAFTStereo,
    "raft-hp": HPRAFTStereo,
    "raft-hp-group": GroupHPRAFTStereo,
}
