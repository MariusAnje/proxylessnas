# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

from data_providers.base_provider import *
import os

class CIFARDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        train_transforms = self.build_train_transform(distort_color, resize_scale)
        test_transforms = transforms.Compose([transforms.ToTensor(),self.normalize])
        train_dataset = torchvision.datasets.CIFAR10(root=self.train_path, train=True, download=False, transform=train_transforms)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            train_indexes, valid_indexes = self.random_sample_valid_set(
                train_dataset.targets, valid_size, self.n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

        
            valid_dataset = torchvision.datasets.CIFAR10(root=self.train_path, train=True, download=False, transform=test_transforms)

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=self.valid_path, train=False, download=False, transform=test_transforms), 
            batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            if os.name == "nt":
                self._save_path = "~/testCode/data"
            else:
                self._save_path = "~/Private/data/CIFAR10"
            # self._save_path = '/dataset/CIFAR10'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path)

    @property
    def valid_path(self):
        return os.path.join(self._save_path)

    @property
    def normalize(self):
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms

    @property
    def resize_value(self):
        return 32

    @property
    def image_size(self):
        return 32

if __name__ == "__main__":
    a = CIFARDataProvider()
    print(a.train.next())