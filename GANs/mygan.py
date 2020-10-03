# using this script to investigate how the gan framework can overfit one image, instead of a whole dataset
# this script is based on `deepclustering2` package.
from pathlib import Path
from typing import Union, Callable

import numpy as np
import torch
from PIL import Image
from torch import nn, Tensor
from torch import optim
from torchvision.utils import make_grid

from deepclustering2.decorator import TikTok
from deepclustering2.epoch._epocher import _Epocher
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter, StorageIncomeDict
from deepclustering2.models import Model
from deepclustering2.trainer import Trainer as _Trainer
from deepclustering2.type import T_optim, T_iter


class GANEpocher(_Epocher):
    y_real_ = 1
    y_fake_ = 0

    def init(self, discriminator: Union[Model, nn.Module], model_optimizer: T_optim, dis_optimizer: T_optim,
             train_iter: T_iter, fixed_noise=None, noise_generator=Callable[[Tensor], Tensor]):
        self._discriminator = discriminator

        self._model_optimizer = model_optimizer
        self._disc_optimizer = dis_optimizer

        self._train_iter = train_iter
        assert fixed_noise is not None, fixed_noise
        self._fixed_noise = fixed_noise
        assert callable(noise_generator), noise_generator
        self._noise_call = noise_generator
        self._bce_criterion = nn.BCELoss()

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        self._discriminator.train()
        assert self._model.training and self._discriminator.training
        fetch_tiktok, batch_tiktok = TikTok(), TikTok()
        report_dict = EpochResultDict()
        for i, (img, _) in zip(self._indicator, self._train_iter):
            fetch_tiktok.tik()
            self.meters["data_fetch_time"].add(fetch_tiktok.cost)
            b, *chw = img.shape
            # train_discriminator
            self._disc_optimizer.zero_grad()
            img = img.to(self._device)
            z_ = self._noise_call(b).to(self._device)
            fake_img = self._model(z_)
            true_target_ = torch.zeros(b, device=self.device).fill_(self.y_real_)
            fake_target_ = torch.zeros(b, device=self.device).fill_(self.y_fake_)
            true_predict = self._discriminator(img).squeeze()
            D_real_loss = self._bce_criterion(true_predict, true_target_)
            self.meters["D(x)"].add(true_predict.mean().item())
            fake_predict = self._discriminator(fake_img.detach()).squeeze()
            self.meters["D(G(z))"].add(fake_predict.mean().item())
            D_fake_loss = self._bce_criterion(fake_predict, fake_target_)
            disc_loss = D_fake_loss + D_real_loss
            disc_loss.backward()
            self._disc_optimizer.step()

            # train generator
            self._model_optimizer.zero_grad()
            # todo: check if this is the most important change
            fake_predict = self._discriminator(fake_img).squeeze()
            gen_loss = self._bce_criterion(fake_predict, true_target_)
            gen_loss.backward()
            self._model_optimizer.step()
            self.meters["D(G(z))2"].add(fake_predict.mean().item())

            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix(report_dict)

            self.meters["generator_loss"].add(gen_loss.item())
            self.meters["discriminator_loss"].add(disc_loss.item())
            self.meters["batch_time"].add(batch_tiktok.cost)
            batch_tiktok.tik()
            fetch_tiktok.tik()
        return report_dict

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(GANEpocher, self)._configure_meters(meters)
        meters.register_meter("generator_loss", AverageValueMeter())
        meters.register_meter("discriminator_loss", AverageValueMeter())
        meters.register_meter("batch_time", AverageValueMeter())
        meters.register_meter("data_fetch_time", AverageValueMeter())
        meters.register_meter("D(x)", AverageValueMeter())
        meters.register_meter("D(G(z))", AverageValueMeter())
        meters.register_meter("D(G(z))2", AverageValueMeter())
        return meters

    def image_writer(self, fixed_noise: bool = True, folder_name="inference"):
        z_ = self._noise_call(self._fixed_noise.shape[0]) if not fixed_noise else self._fixed_noise
        self._model.eval()
        with torch.no_grad():
            images = self._model(z_)

        grid_image = make_grid([*images], padding=4, nrow=8).cpu().numpy().transpose(1, 2, 0)
        grid_image = (grid_image + 1.0) / 2.0
        folder = Path(self.trainer._save_dir, folder_name)
        folder.mkdir(exist_ok=True)
        file_name = f"Epoch_{self._cur_epoch:03d}.png"
        Image.fromarray((grid_image * 255).astype(np.uint8)).save(str(folder / file_name))
        return grid_image


class GANTrainer(_Trainer):
    RUN_PATH = "./runs"

    def __init__(self, model: nn.Module, discriminator: nn.Module, train_iter: T_iter, save_dir: str = "base",
                 max_epoch: int = 100, num_batches: int = 100, device: str = "cpu", configuration=None):
        super().__init__(model, save_dir, max_epoch, num_batches, device, configuration)
        self._discriminator = discriminator

        self._model_optimizer = optim.Adam(self._model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self._disc_optimizer = optim.Adam(self._discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self._train_iter = train_iter

    def init(self):
        self._noise_generator = lambda b: torch.randn((b, 100), device=self._device).view(-1, 100, 1, 1)
        self._fixed_noise = self._noise_generator(64)

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        epocher = GANEpocher(self._model, self._num_batches, self._cur_epoch, self._device)
        epocher.set_trainer(self)
        epocher.init(self._discriminator, self._model_optimizer, self._disc_optimizer, self._train_iter,
                     fixed_noise=self._fixed_noise, noise_generator=self._noise_generator)
        result_dict = epocher.run()
        grid_image = epocher.image_writer(fixed_noise=True)
        self._writer.add_image("visual", grid_image.transpose(2, 0, 1), global_step=self._cur_epoch)
        return result_dict

    def _start_training(self, *args, **kwargs):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            storage_per_epoch = StorageIncomeDict(tra=train_result, )
            self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict=storage_per_epoch, epoch=self._cur_epoch)
            # save_checkpoint
            # self.save(cur_score)
            # save storage result on csv file.
            self._save_to("last.pth")
            self._storage.to_csv(self._save_dir)


if __name__ == '__main__':
    from torchvision import transforms, datasets
    from deepclustering2.dataloader.sampler import InfiniteRandomSampler
    from torch.utils.data import DataLoader, Subset #noqa
    import arch

    batch_size = 64

    # data_loader
    img_size = 64
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(size=img_size, scale=(0.7, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5, 0.5), std=(0.5,0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    # train_dataset = Subset(train_dataset, np.where(train_dataset.targets == 5)[0][:5])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=InfiniteRandomSampler(train_dataset, shuffle=True),  # noqa
        num_workers=4
    )
    G = arch.OfficialGenerator(100, 32, 3)
    D = arch.OfficialDiscriminator(3, 64)
    trainer = GANTrainer(G, D, iter(train_loader), save_dir="mygan_cifar", max_epoch=100, num_batches=100,
                         device="cuda", configuration=None)
    trainer.init()
    # trainer.load_state_dict_from_path("runs/tmp/last.pth")
    trainer.start_training()
