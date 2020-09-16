# using this script to investigate how the gan framework can overfit one image, instead of a whole dataset
# this script is based on `deepclustering2` package.
from typing import Union

from deepclustering2.epoch._epocher import _Epocher
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.models import Model
from deepclustering2.trainer.trainer import T_optim
from torch import nn


class GANEpocher(_Epocher):
    def __init__(self, model: Union[Model, nn.Module], discriminator: Union[Model, nn.Module], cur_epoch=0,
                 device="cpu") -> None:
        super().__init__(model, cur_epoch, device)
        self._discriminator = discriminator

    def init(self, model_optimizer: T_optim, dis_optimizer: T_optim):
        self._model_optimizer = model_optimizer
        self._discriminator = dis_optimizer

    def _run(self, *args, **kwargs) -> EpochResultDict:
        pass
