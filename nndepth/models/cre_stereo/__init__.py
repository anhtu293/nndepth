from .model import CREStereoBase
from .loss import CREStereoLoss
from .cre_trainer import CREStereoTrainer


STEREO_MODELS: dict[str, type[CREStereoBase]] = {
    "base": CREStereoBase,
}
