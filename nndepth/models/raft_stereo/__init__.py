from .model import BaseRAFTStereo, Coarse2FineGroupRepViTRAFTStereo
from .loss import RAFTLoss
from .raft_trainer import RAFTTrainer

from nndepth.utils.base_model import BaseModel

STEREO_MODELS: dict[str, BaseModel] = {
    "base": BaseRAFTStereo,
    "coarse2fine": Coarse2FineGroupRepViTRAFTStereo,
}
