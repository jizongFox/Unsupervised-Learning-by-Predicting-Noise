from copy import deepcopy
from typing import Tuple, Union

from deepclustering2.epoch._epocher import _Epocher  # noqa
from deepclustering2.meters2 import MeterInterface, AverageValueMeter, EpochResultDict
from deepclustering2.models import Model
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer import Trainer as _Trainer
from deepclustering2.trainer.trainer import T_optim, T_loader
from torch import nn
from torch.nn.functional import normalize

from _utils import FeatureExtractor, is_normalize, calc_optimal_target_permutation


class PreTrainEpocher(_Epocher):

    def init(self, optimizer: T_optim, train_loader: T_loader, num_batches: int = 1000, permutation_step=10):
        self._optimizer = optimizer
        self._train_iter = train_loader
        self._permutation_interval = permutation_step  # noqa
        self._num_batches = num_batches  # noqa
        self._l2_criterion = nn.MSELoss()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(PreTrainEpocher, self)._configure_meters(meters)
        meters.register_meter("loss", AverageValueMeter())
        return meters

    def _run(
        self, *args, **kwargs
    ) -> EpochResultDict:
        self._model.train()
        report_dict = EpochResultDict()
        with FeatureExtractor(self._model, feature_names="fc") as self._fexactor:  # noqa
            with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
                for batch_num, (data, _, nat, indices) in zip(indicator, self._train_iter):
                    loss = self._batch_run(batch_num, data, nat, indices)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    self.meters["loss"].add(loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict

    def _batch_run(self, batch_num, data, nat, indices):
        data = data.to(self._device)
        _ = self._model(data)
        embedding = normalize(self._fexactor["fc"])
        loss = self._l2_criterion(embedding, nat)
        self._permutate_target(batch_num, embedding, nat, indices)
        return loss

    def _permutate_target(self, batch_num, embedding, targets, indices) -> None:
        if (batch_num + 1) % self._permutation_interval == 0:
            assert is_normalize(embedding) and is_normalize(targets)
            new_target = calc_optimal_target_permutation(embedding, targets)
            self._train_iter._dataset.update_nat(indices, new_target)


class TrainClassifierEpocher(PreTrainEpocher):

    def init(self, optimizer: T_optim, train_loader: T_loader, num_batches: int = 1000):
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._num_batches = num_batches
        self._ce_criterion = nn.CrossEntropyLoss()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("acc", AverageValueMeter())
        return meters

    def _run(
        self, *args, **kwargs
    ) -> Tuple[EpochResultDict, float]:
        self._model.train()
        report_dict = EpochResultDict()
        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for batch_num, (data, target, _, _) in zip(indicator, self._train_iter):
                loss = self._batch_run(batch_num, data, target)
                loss.backward()
                self.meters["loss"].add(loss.item())
                report_dict = self.meters.tracking_status()
                indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["acc"].summary()["acc"]

    def _batch_run(self, batch_num, data, target):  # noqa
        data, target = data.to(self._device), target.to(self._device)
        predict = self._model(data)
        loss = self._ce_criterion(predict, target)
        self.meters["acc"].add(predict.max(1)[1], target)
        return loss


class EvalEpocher(TrainClassifierEpocher):

    def init(self, val_loader):
        self._val_loader = val_loader
        self._ce_criterion = nn.CrossEntropyLoss()

    def _run(
        self, *args, **kwargs
    ) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        report_dict = EpochResultDict()
        with tqdm(range(len(self._val_loader))).set_desc_from_epocher(self) as indicator:
            for batch_num, (data, target, _, _) in zip(indicator, self._val_loader):
                loss = self._batch_run(batch_num, data, target)
                loss.backward()
                self.meters["loss"].add(loss.item())
                report_dict = self.meters.tracking_status()
                indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["acc"].summary()["acc"]


class Trainer(_Trainer):
    def __init__(self, model: Union[Model, nn.Module], max_epoch_pretrain=100, max_epoch_finetune=100,
                 save_dir: str = "base", num_batches: int = 100, device: str = "cpu",
                 configuration=None):
        super().__init__(model, save_dir, 0, num_batches, device, configuration)
        del self._max_epoch
        self._max_epoch_pretrain = max_epoch_pretrain
        self._max_epoch_finetune = max_epoch_finetune

    def init_pretrain(self):
        config = deepcopy(self._config)

    def init_finetune(self):
        config = deepcopy(self._config)
