from .model import CREStereoBase
from .loss import CREStereoLoss
from .cre_trainer import CREStereoTrainer

from nndepth.utils.base_model import BaseModel


STEREO_MODELS: dict[str, BaseModel] = {
    "base": CREStereoBase,
}
