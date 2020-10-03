import torchvision.datasets as dset
from torchvision import transforms


def create_dataset(dataset_name: str, dataset_root: str, img_size: int, ):
    if dataset_name in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=dataset_root,
                                   transform=transforms.Compose([
                                       transforms.Resize(img_size),
                                       transforms.CenterCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nc = 3
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(root=dataset_root, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 3

    elif dataset_name == 'mnist':
        dataset = dset.MNIST(root=dataset_root, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
        nc = 1

    elif dataset_name == 'fake':
        dataset = dset.FakeData(image_size=(3, img_size, img_size),
                                transform=transforms.ToTensor())
        nc = 3
    else:
        raise NotImplementedError(dataset_name)
    return dataset, nc
