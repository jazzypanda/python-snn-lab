import os
import subprocess
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def check_and_prepare_imagenet(data_dir, target_dir=None):
    """
    Checks if ImageNet is extracted in target_dir. 
    If not, and archives exist in data_dir, extracts them to target_dir.
    Returns the path to the ready-to-use dataset.
    """
    # If no target_dir is specified, default to data_dir (assuming it's writable/extracted)
    if target_dir is None:
        target_dir = data_dir

    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    # Check if already extracted
    if os.path.isdir(train_dir):
        print(f"ImageNet train data found at {train_dir}")
        return target_dir
    
    print(f"ImageNet train data not found at {train_dir}. Checking for archives in {data_dir}...")
    
    # Check for archives in data_dir (or ILSVRC2012 subdir)
    archive_root = data_dir
    if os.path.isdir(os.path.join(data_dir, 'ILSVRC2012')):
        archive_root = os.path.join(data_dir, 'ILSVRC2012')
        
    train_tar = os.path.join(archive_root, 'ILSVRC2012_img_train.tar')
    val_tar = os.path.join(archive_root, 'ILSVRC2012_img_val.tar')
    
    if not os.path.isfile(train_tar):
        # Fallback: If we can't find archives and can't find extracted folders, just return 
        # the original path and let ImageFolder raise the error (or try to find nested structure)
        return data_dir
        
    # If we are here, archives exist, but extracted folders don't.
    # We MUST have a writable target_dir.
    if target_dir == data_dir:
        print("WARNING: Archives found but no extracted folders, and no separate --extract-root provided.")
        print("Attempting to extract in-place (this may fail if read-only)...")
    
    print(f"Extracting ImageNet to {target_dir} (This may take a while)...")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 1. Extract Train
    print(f"Extracting {train_tar}...")
    subprocess.check_call(['tar', '-xf', train_tar, '-C', train_dir])
    
    print("Extracting training sub-archives...")
    # Find all .tar files in train_dir and extract them to their own folders
    # We use a shell snippet for performance
    cmd = (
        f"find {train_dir} -name '*.tar' | while read NAME ; do "
        f"mkdir -p \"${{NAME%.tar}}\"; "
        f"tar -xf \"${{NAME}}\" -C \"${{NAME%.tar}}\"; "
        f"rm -f \"${{NAME}}\"; "
        f"done"
    )
    subprocess.check_call(cmd, shell=True)
    
    # 2. Extract Val
    if os.path.isfile(val_tar):
        print(f"Extracting {val_tar}...")
        subprocess.check_call(['tar', '-xf', val_tar, '-C', val_dir])
        print("Note: Validation set extracted flat. You may need to run a val-prep script to organize into class folders for ImageFolder.")
        
        # Optional: Try to organize val if the devkit or script is available? 
        # For now, we leave it flat to avoid complex dependencies, but warn the user.
    
    print("Extraction complete.")
    return target_dir

def get_imagenet_loaders(data_dir, batch_size=64, num_workers=8, distributed=False, extract_root=None):
    
    # Pre-check / Extract
    # If extract_root is provided, we use it as the target for extraction/loading
    real_data_path = check_and_prepare_imagenet(data_dir, extract_root)
    
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
    train_dir = os.path.join(real_data_path, 'train')
    val_dir = os.path.join(real_data_path, 'val')

    # If validation directory is flat (no class subfolders), ImageFolder will fail or treat images as classes.
    # We add a safeguard: if val_dir exists but has no subdirs, we might skip validation or warn.
    
    train_set = torchvision.datasets.ImageFolder(
        root=train_dir, transform=transform_train)
    
    # Robust loading for val
    try:
        test_set = torchvision.datasets.ImageFolder(
            root=val_dir, transform=transform_test)
    except Exception as e:
        print(f"Warning: Could not create Validation Dataset from {val_dir}. Error: {e}")
        print("Validation will be skipped. Ensure validation images are organized into class folders.")
        test_set = None

    train_sampler = DistributedSampler(train_set) if distributed else None
    
    if test_set:
        test_sampler = DistributedSampler(test_set, shuffle=False) if distributed else None
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, sampler=test_sampler, persistent_workers=True)
    else:
        test_loader = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)

    return train_loader, test_loader, train_sampler
