from .model import BaseRAFTStereo, Coarse2FineGroupRepViTRAFTStereo
from .loss import RAFTLoss
from .raft_trainer import RAFTTrainer


STEREO_MODELS: dict[str, type[BaseRAFTStereo] | type[Coarse2FineGroupRepViTRAFTStereo]] = {
    "base": BaseRAFTStereo,
    "coarse2fine": Coarse2FineGroupRepViTRAFTStereo,
}
