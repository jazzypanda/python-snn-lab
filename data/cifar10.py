import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_cifar10_loaders(data_dir='./data', batch_size=128, num_workers=4, distributed=False):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    train_sampler = DistributedSampler(train_set) if distributed else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=test_sampler)

    return train_loader, test_loader, train_sampler
