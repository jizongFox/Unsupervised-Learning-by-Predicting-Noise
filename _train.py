import os
from typing import Tuple, Union

import torch
from deepclustering2.epoch._epocher import _Epocher  # noqa
from deepclustering2.meters2 import MeterInterface, AverageValueMeter, EpochResultDict, StorageIncomeDict, Storage, \
    ConfusionMatrix
from deepclustering2.models import Model
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.schedulers.warmup_scheduler import GradualWarmupScheduler
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer import Trainer as _Trainer
from deepclustering2.trainer.trainer import T_optim, T_loader
from torch import nn
from torch.nn.functional import normalize

from _utils import FeatureExtractor, is_normalize, calc_optimal_target_permutation, backbone_parameters, \
    classifier_parameters


class PreTrainEpocher(_Epocher):

    def init(self, optimizer: T_optim, train_loader: T_loader, num_batches: int = 1000, permutation_step=10):
        self._optimizer = optimizer
        self._train_iter = train_loader
        self._permutation_interval = permutation_step  # noqa
        self._num_batches = num_batches  # noqa
        self._l2_criterion = nn.MSELoss()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(PreTrainEpocher, self)._configure_meters(meters)
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("loss", AverageValueMeter())
        return meters

    def _run(
        self, *args, **kwargs
    ) -> EpochResultDict:
        self._model.train()
        report_dict = EpochResultDict()
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        with FeatureExtractor(self._model, feature_names="avgpool") as self._fexactor:  # noqa
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
        data, nat = data.to(self._device), nat.to(self._device)
        _ = self._model(data)
        embedding = normalize(self._fexactor["avgpool"],).squeeze()
        loss = self._l2_criterion(embedding, nat)
        self._permutate_target(batch_num, embedding, nat, indices)
        return loss

    def _permutate_target(self, batch_num, embedding, targets, indices) -> None:
        assert is_normalize(embedding) and is_normalize(targets)
        if (batch_num + 1) % self._permutation_interval == 0:
            new_targets = calc_optimal_target_permutation(embedding, targets)
            self._train_iter._dataset.update_nat(indices, new_targets)


class TrainClassifierEpocher(PreTrainEpocher):

    def init(self, optimizer: T_optim, train_loader: T_loader, num_batches: int = 1000):
        self._optimizer = optimizer
        self._train_iter = train_loader
        self._num_batches = num_batches
        self._ce_criterion = nn.CrossEntropyLoss()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("acc", ConfusionMatrix(num_classes=10))
        return meters

    def _run(
        self, *args, **kwargs
    ) -> EpochResultDict:
        self._model.train()
        report_dict = EpochResultDict()
        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for batch_num, (data, target, _, _) in zip(indicator, self._train_iter):
                loss = self._batch_run(batch_num, data, target)
                loss.backward()
                self.meters["loss"].add(loss.item())
                report_dict = self.meters.tracking_status()
                indicator.set_postfix_dict(report_dict)
        return report_dict

    def _batch_run(self, batch_num, data, target):  # noqa
        data, target = data.to(self._device), target.to(self._device)
        predict = self._model(data)
        loss = self._ce_criterion(predict, target)
        self.meters["acc"].add(predict.max(1)[1], target)
        return loss


class EvalEpocher(TrainClassifierEpocher):

    def init(self, val_loader):
        self._val_loader = val_loader  # noqa
        self._ce_criterion = nn.CrossEntropyLoss()  # noqa

    def _run(
        self, *args, **kwargs
    ) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        report_dict = EpochResultDict()
        with tqdm(range(len(self._val_loader))).set_desc_from_epocher(self) as indicator:
            for batch_num, (data, target, _, _) in zip(indicator, self._val_loader):
                loss = self._batch_run(batch_num, data, target)
                self.meters["loss"].add(loss.item())
                report_dict = self.meters.tracking_status()
                indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["acc"].summary()["acc"]


class Trainer(_Trainer):
    RUN_PATH = "./runs"

    def __init__(self, model: Union[Model, nn.Module], pretrain_loader: T_loader, finetune_loader: T_loader,
                 val_loader: T_loader, max_epoch_pretrain=100, max_epoch_finetune=100,
                 save_dir: str = "base", num_batches: int = 100, device: str = "cpu",
                 configuration=None):
        super().__init__(model, save_dir, 0, num_batches, device, configuration)
        del self._max_epoch
        self._pretrain_loader = pretrain_loader
        self._finetune_loader = finetune_loader
        self._val_loader = val_loader
        self._max_epoch_pretrain = max_epoch_pretrain
        self._max_epoch_finetune = max_epoch_finetune

        self._pretrain_storage = Storage()
        self._finetune_storage = Storage()

    def init_pretrain(self):
        self._optimizer = torch.optim.Adam(backbone_parameters(self._model), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=self._max_epoch_pretrain - 10, eta_min=1e-5
        )
        self._scheduler = GradualWarmupScheduler(self._optimizer, 3000, 10, after_scheduler=scheduler)

    def run_pretrain(self):
        self.to(self._device)

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_pretrain):
            pretrain_epocher = PreTrainEpocher(
                model=self._model, cur_epoch=self._cur_epoch, device=self._device
            )
            pretrain_epocher.init(
                self._optimizer, self._pretrain_loader, num_batches=self._num_batches, permutation_step=50
            )
            pretrain_dict = pretrain_epocher.run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(PRETRAIN_ENCODER=pretrain_dict, )
            self._pretrain_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain"))

    def init_finetune(self):
        self._optimizer = torch.optim.Adam(classifier_parameters(self._model), lr=1e-5, weight_decay=1e-4)  # noqa
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=self._max_epoch_pretrain - 10, eta_min=1e-5
        )
        self._scheduler = GradualWarmupScheduler(self._optimizer, 3000, 10, after_scheduler=scheduler)  # noqa

    def run_finetune(self):
        self.to(self._device)

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_finetune):
            finetune_epocher = TrainClassifierEpocher(
                model=self._model, cur_epoch=self._cur_epoch, device=self._device
            )
            finetune_epocher.init(
                self._optimizer, self._pretrain_loader, num_batches=self._num_batches,
            )
            pretrain_dict = finetune_epocher.run()
            with torch.no_grad():
                val_epocher = EvalEpocher(model=self._model, cur_epoch=self._cur_epoch, device=self._device)
                val_epocher.init(self._val_loader)
                val_dict, cur_score = val_epocher.run()

            self._scheduler.step()
            storage_dict = StorageIncomeDict(FINETUNE=pretrain_dict, EVAL=val_dict)
            self._finetune_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self.save(cur_score)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "finetune"))

    def _start_training(self, *args, **kwargs):
        self.init_pretrain()
        self.run_pretrain()
        self.init_finetune()
        self.run_finetune()
