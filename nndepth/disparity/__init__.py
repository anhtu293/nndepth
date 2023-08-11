from nndepth.disparity.models import CREStereo, IGEVStereoMBNet, BaseRAFTStereo, HPRAFTStereo


MODELS = {
    "crestereo-base": CREStereo,
    "igev-mbnet": IGEVStereoMBNet,
    "raft-base": BaseRAFTStereo,
    "raft-hp": HPRAFTStereo
}
