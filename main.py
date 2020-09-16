from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from torch.utils.data import DataLoader

from _train import Trainer
from _utils import NatCIFAR10, train_transform, test_transform, resnet34

network = resnet34(input_dim=3, num_classes=10)

train_set = NatCIFAR10(root="./.data", train=True, z_dims=512, transform=train_transform, download=True)
test_set = NatCIFAR10(root="./.data", train=False, z_dims=512, transform=test_transform, download=True)

train_loader = DataLoader(train_set, num_workers=2, batch_size=256,
                          sampler=InfiniteRandomSampler(train_set, shuffle=True))
test_loader = DataLoader(test_set, batch_size=100, num_workers=2)

trainer = Trainer(network, iter(train_loader), iter(train_loader), test_loader, max_epoch_pretrain=100,
                  max_epoch_finetune=100, num_batches=256, save_dir="tmp3", device="cuda")
trainer.start_training()
