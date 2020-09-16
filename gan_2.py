# using this script to investigate how the gan framework can overfit one image, instead of a whole dataset
# this script is based on `deepclustering2` package.
from pathlib import Path
from typing import Union, Callable

import numpy as np
import torch
from PIL import Image
from torch import nn, Tensor
from torch import optim
from torch.nn import functional as F
from torchvision.utils import make_grid

from deepclustering2.decorator import TikTok
from deepclustering2.epoch._epocher import _Epocher
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter, StorageIncomeDict
from deepclustering2.models import Model
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer import Trainer as _Trainer
from deepclustering2.type import T_optim, T_iter


# %% network definition

class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        self.weight_init(0.0, 0.02)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)
        self.weight_init(0.0, 0.02)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()


# %% epocher definition

class GANEpocher(_Epocher):
    y_real_ = 1
    y_fake_ = 0
    indicator: tqdm

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
        datafetch_tiktok, batch_run_tiktok = TikTok(), TikTok()
        report_dict = EpochResultDict()
        for i, (img, _) in zip(self._indicator, self._train_iter):
            datafetch_tiktok.tik()
            self.meters["data_fetch_time"].add(datafetch_tiktok.cost)
            b, *chw = img.shape
            # train_discriminator
            self._disc_optimizer.zero_grad()
            img = img.to(self._device)
            z_ = self._noise_call(b).to(self._device)
            fake_img = self._model(z_)
            all_imgs = torch.cat([img, fake_img.detach()], dim=0)
            predict = self._discriminator(all_imgs).squeeze()
            true_target_ = torch.zeros(b, device=self.device).fill_(self.y_real_)
            fake_target_ = torch.zeros(b, device=self.device).fill_(self.y_fake_)
            disc_loss = self._bce_criterion(predict, torch.cat((true_target_, fake_target_), dim=0))
            disc_loss.backward()
            self._disc_optimizer.step()
            self.meters["discriminator_loss"].add(disc_loss.item())
            # train generator
            self._model_optimizer.zero_grad()
            z_ = self._noise_call(b).to(self._device)
            fake_img = self._model(z_)
            fake_predict = self._discriminator(fake_img).squeeze()
            gen_loss = self._bce_criterion(fake_predict, true_target_)
            gen_loss.backward()
            self._model_optimizer.step()
            self.meters["generator_loss"].add(gen_loss.item())
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix(report_dict)
            datafetch_tiktok.tik()
            batch_run_tiktok.tik()
            self.meters["batch_time"].add(batch_run_tiktok.cost)
        return report_dict

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(GANEpocher, self)._configure_meters(meters)
        meters.register_meter("generator_loss", AverageValueMeter())
        meters.register_meter("discriminator_loss", AverageValueMeter())
        meters.register_meter("batch_time", AverageValueMeter())
        meters.register_meter("data_fetch_time", AverageValueMeter())
        return meters

    def image_writer(self, fixed_noise: bool = True, folder_name="inference"):
        z_ = self._noise_call(self._fixed_noise.shape[0]) if not fixed_noise else self._fixed_noise
        self._model.eval()
        with torch.no_grad():
            images = self._model(z_.to(self._device))

        grid_image = make_grid([*images], padding=4, nrow=8, normalize=True).cpu().numpy().transpose(1, 2, 0)
        folder = Path(self.trainer._save_dir, folder_name)
        folder.mkdir(exist_ok=True)
        file_name = f"Epoch_{self._cur_epoch:03d}.png"
        Image.fromarray((grid_image * 255).astype(np.uint8)).save(str(folder / file_name))


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
        self._noise_generator = lambda b: torch.randn((b, 100)).view(-1, 100, 1, 1)
        self._fixed_noise = self._noise_generator(64)

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        epocher = GANEpocher(self._model, self._num_batches, self._cur_epoch, self._device)
        epocher.set_trainer(self)
        epocher.init(self._discriminator, self._model_optimizer, self._disc_optimizer, self._train_iter,
                     fixed_noise=self._fixed_noise, noise_generator=self._noise_generator)
        result_dict = epocher.run()
        epocher.image_writer(fixed_noise=True)
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
            self._storage.to_csv(self._save_dir)


if __name__ == '__main__':
    from torchvision import transforms, datasets
    from deepclustering2.dataloader.sampler import InfiniteRandomSampler
    from torch.utils.data import DataLoader

    batch_size = 128

    # data_loader
    img_size = 64
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=InfiniteRandomSampler(train_dataset, shuffle=True), num_workers=4
    )
    G = generator(128)
    D = discriminator(128)
    trainer = GANTrainer(G, D, iter(train_loader), save_dir="gan_version1", max_epoch=30, num_batches=len(train_loader),
                         device="cuda", configuration=None)
    trainer.init()
    trainer.start_training()
