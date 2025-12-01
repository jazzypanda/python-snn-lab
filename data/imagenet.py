import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_imagenet_loaders(data_dir, batch_size=64, num_workers=8, distributed=False):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Assumes standard ImageNet structure: train/ and val/ folders
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.isdir(train_dir):
        # Try looking into ILSVRC2012 subdirectory (common in AutoDL)
        potential_dir = os.path.join(data_dir, 'ILSVRC2012')
        if os.path.isdir(os.path.join(potential_dir, 'train')):
            print(f"Detected ImageNet structure inside ILSVRC2012. Updating paths...")
            train_dir = os.path.join(potential_dir, 'train')
            val_dir = os.path.join(potential_dir, 'val')

    train_set = torchvision.datasets.ImageFolder(
        root=train_dir, transform=transform_train)
    
    test_set = torchvision.datasets.ImageFolder(
        root=val_dir, transform=transform_test)

    train_sampler = DistributedSampler(train_set) if distributed else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=test_sampler, persistent_workers=True)

    return train_loader, test_loader, train_sampler
