from .model import IGEVStereoMBNet
from .loss import IGEVStereoLoss
from .igev_trainer import IGEVStereoTrainer


STEREO_MODELS: dict[str, type[IGEVStereoMBNet]] = {
    "igev_stereo_mbnet": IGEVStereoMBNet,
}
