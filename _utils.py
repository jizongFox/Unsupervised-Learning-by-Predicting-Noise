from collections import OrderedDict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet
from torchvision.models.resnet import conv1x1, Bottleneck, BasicBlock
from torchvision.transforms import Compose

__all__ = ["is_normalize", "generate_random_targets", "train_transform", "test_transform", "NatCIFAR10",
           "backbone_parameters", "calc_optimal_target_permutation", "FeatureExtractor", "resnet34"]


def is_normalize(tensor: Tensor):
    b, *d = tensor.shape
    tensor = tensor.view(b, -1)
    return torch.all(tensor.norm(dim=1, p=2) == 1.0)


def generate_random_targets(n: int, z: int):
    """
    Generate a matrix of random target assignment.
    Each target assignment vector has unit length (hence can be view as random point on hypersphere)
    :param n: the number of samples to generate.
    :param z: the latent space dimensionality
    :return: the sampled representations
    """

    # Generate random targets using gaussian distrib.
    samples = np.random.normal(0, 1, (n, z)).astype(np.float32)
    # rescale such that fit on unit sphere.
    radiuses = np.expand_dims(np.sqrt(np.sum(np.square(samples), axis=1)), 1)
    # return rescaled targets
    return samples / radiuses


train_transform = Compose([
    transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
    transforms.ToTensor()
])
test_transform = Compose([
    transforms.CenterCrop(28),
    transforms.ToTensor()
])


class NatCIFAR10(CIFAR10):
    def __init__(self, root, train: bool = True, z_dims: int = 50, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train, transform, target_transform, download)
        if train:
            self._nat = generate_random_targets(len(self.data), z_dims)
        else:
            self._nat = np.empty(shape=(len(self.data), z_dims))

    def update_nat(self, idx, target):
        for i, t in zip(idx, target):
            self._nat[i] = t

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        nat = self._nat[index]
        return img, target, nat, index


def backbone_parameters(network):
    """
    return the backbone parameters of a resnet network to train the encoder
    :param network: resnet
    :return: parameter iterator
    """
    new_named_parameter_dict = {k: v for k, v in network.named_parameters() if k != "fc"}

    for gen in new_named_parameter_dict:
        yield gen


def calc_optimal_target_permutation(feats: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute the new target assignment that minimises the SSE between the mini-batch feature space and the targets.
    :param feats: the learnt features (given some input images)
    :param targets: the currently assigned targets.
    :return: the targets reassigned such that the SSE between features and targets is minimised for the batch.
    """
    # Compute cost matrix
    cost_matrix = np.zeros([feats.shape[0], targets.shape[0]])
    # calc SSE between all features and targets
    for i in range(feats.shape[0]):
        cost_matrix[:, i] = np.sum(np.square(feats - targets[i, :]), axis=1)

    _, col_ind = linear_sum_assignment(cost_matrix)
    # Permute the targets based on hungarian algorithm optimisation
    targets[range(feats.shape[0])] = targets[col_ind]
    return targets


class FeatureExtractor(nn.Module):
    class _FeatureExtractor:
        def __call__(self, _, input, result):
            self.feature = result

    def __init__(self, net: ResNet, feature_names="fc") -> None:
        super().__init__()
        self._net = net
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names

    def __enter__(self):
        self._feature_exactors = OrderedDict()
        self._hook_handlers = OrderedDict()
        for f in self._feature_names:
            extractor = self._FeatureExtractor()
            handler = getattr(self._net, f).register_forward_hook(extractor)
            self._feature_exactors[f] = extractor
            self._hook_handlers[f] = handler
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self._hook_handlers.items():
            v.remove()
        del self._feature_exactors, self._hook_handlers

    def __getitem__(self, item):
        if item in self._feature_exactors:
            return self._feature_exactors[item].feature
        return super().__getitem__(item)

    def __iter__(self):
        for k, v in self._feature_exactors.items():
            yield v.feature


class ResNet(nn.Module):

    def __init__(self, block, layers, input_dim=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.input_dim = input_dim
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_dim, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
