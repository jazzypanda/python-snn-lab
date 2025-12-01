import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils.distributed as utils
from utils.metrics import MetricLogger, accuracy
from data.cifar10 import get_cifar10_loaders
from data.imagenet import get_imagenet_loaders
from models.resnet import ResNet18

def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch SNN Training (BF16 Optimized)')
    parser.add_argument('--data-path', default='./data', type=str, help='dataset path')
    parser.add_argument('--extract-root', default=None, type=str, help='extraction path for imagenet')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument('--model', default='resnet18', type=str, help='model name')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--T', default=4, type=int, help='simulation time steps')
    parser.add_argument('--tau', default=2.0, type=float, help='membrane time constant')
    parser.add_argument('--alpha', default=4.0, type=float, help='surrogate gradient alpha')
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--test-only', action='store_true')
    
    # Distributed training parameters
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--dist-url', default='env://')
    
    return parser

def main(args):
    utils.init_distributed_mode(args)
    
    # Ampere Optimization
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda')

    # Data loaders
    print(f"Loading data from {args.data_path}")
    if args.dataset == 'cifar10':
        num_classes = 10
        train_loader, test_loader, train_sampler = get_cifar10_loaders(
            args.data_path, args.batch_size, args.workers, args.distributed)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_loader, test_loader, train_sampler = get_imagenet_loaders(
            args.data_path, args.batch_size, args.workers, args.distributed, args.extract_root)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Model
    print(f"Creating model: {args.model}")
    neuron_params = {'tau': args.tau, 'surrogate_alpha': args.alpha, 'detach_reset': True}
    
    # We only support ResNet18 in this high-perf branch
    if args.model == 'resnet18':
        model = ResNet18(T=args.T, num_classes=num_classes, neuron_params=neuron_params)
    else:
        # Fallback or Error? User might still type resnet19.
        # Since we removed ResNet19 class, this would crash. 
        # But we can just map it or error out.
        print(f"Warning: Model {args.model} not explicitly supported in BF16 branch. Defaulting to ResNet18 structure.")
        model = ResNet18(T=args.T, num_classes=num_classes, neuron_params=neuron_params)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Optimizer
    print(f"Using SGD: lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Scheduler
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5)
    warmup_epochs = 5
    if args.epochs > warmup_epochs:
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Writer
    writer = None
    if utils.is_main_process() and args.output_dir:
        run_name = f"BF16_T{args.T}_tau{args.tau}_bs{args.batch_size}"
        args.output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.output_dir)

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        if test_loader:
            evaluate(model, test_loader, device, 0, writer, args)
        return

    print("Start training (BF16 Mode)")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        train_one_epoch(model, optimizer, train_loader, device, epoch, writer, args)
        lr_scheduler.step()
        
        if test_loader:
            evaluate(model, test_loader, device, epoch, writer, args)
        
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))
            if epoch % 10 == 0:
                 utils.save_on_master(checkpoint, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

    total_time = time.time() - start_time
    print(f"Training time {total_time:.2f}s")

def train_one_epoch(model, optimizer, loader, device, epoch, writer, args):
    model.train()
    logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    
    for i, (image, target) in enumerate(loader):
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        if hasattr(model, 'reset'): model.reset()
        elif hasattr(model.module, 'reset'): model.module.reset()
            
        optimizer.zero_grad()
        
        # BF16 Autocast - No Scaler needed
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(image)
            loss = nn.CrossEntropyLoss()(output, target)
            
        loss.backward()
        # Clip grad is still good practice
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        acc1, = accuracy(output, target, topk=(1,))
        batch_size = image.shape[0]
        logger.update(loss=loss.item(), acc=acc1.item())
        
        if i % 50 == 0 and utils.is_main_process():
            lr = optimizer.param_groups[0]["lr"]
            print(f"{header} [{i}/{len(loader)}] lr: {lr:.6f}  {str(logger)}")
            if writer:
                global_step = epoch * len(loader) + i
                writer.add_scalar('Train/Loss', logger.meters['loss'].avg, global_step)
                writer.add_scalar('Train/Acc', logger.meters['acc'].avg, global_step)

def evaluate(model, loader, device, epoch, writer, args):
    model.eval()
    logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            if hasattr(model, 'reset'): model.reset()
            elif hasattr(model.module, 'reset'): model.module.reset()
            
            # BF16 Inference
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(image)
                loss = nn.CrossEntropyLoss()(output, target)
            
            acc1, = accuracy(output, target, topk=(1,))
            logger.update(loss=loss.item(), acc=acc1.item())

    print(f"{header} {str(logger)}")
    
    if writer and utils.is_main_process():
        writer.add_scalar('Test/Acc', logger.meters['acc'].avg, epoch)
        writer.add_scalar('Test/Loss', logger.meters['loss'].avg, epoch)
        
    return logger.meters['acc'].avg

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)