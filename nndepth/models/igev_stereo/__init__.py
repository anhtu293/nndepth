from .model import IGEVStereoMBNet
from .loss import IGEVStereoLoss
from .igev_trainer import IGEVStereoTrainer

from nndepth.utils.base_model import BaseModel


STEREO_MODELS: dict[str, BaseModel] = {
    "igev_stereo_mbnet": IGEVStereoMBNet,
}
