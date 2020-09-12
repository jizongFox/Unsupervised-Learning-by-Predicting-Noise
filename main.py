from torch.utils.data import DataLoader

from _utils import NatCIFAR10, train_transform, test_transform, resnet34
train_set = NatCIFAR10(root="./.data", train=True, z_dims=50, transform=train_transform, download=True)
test_set = NatCIFAR10(root="./.data", train=False, z_dims=50, transform=test_transform, download=True)

network = resnet34(input_dim=3, num_classes=10)

train_loader = iter(DataLoader(train_set, num_workers=2))

pass
