import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import utils.distributed as utils
from utils.metrics import AverageMeter, accuracy
from utils.monitor import SNNMonitor
from data.cifar10 import get_cifar10_loaders
from data.imagenet import get_imagenet_loaders
from models.resnet import ResNet19, ResNet18

def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch SNN Training')
    parser.add_argument('--data-path', default='./data', type=str, help='dataset path')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet'], help='dataset name')
    parser.add_argument('--model', default='resnet19', type=str, help='model name')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--T', default=4, type=int, help='simulation time steps')
    parser.add_argument('--tau', default=2.0, type=float, help='membrane time constant')
    parser.add_argument('--alpha', default=4.0, type=float, help='surrogate gradient alpha (shape factor)')
    parser.add_argument('--output-dir', default='./output', help='path where to save, empty for no save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--test-only', action='store_true', help='only test model')
    
    # Distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    return parser

def main(args):
    utils.init_distributed_mode(args)
    
    # 1. Enable cuDNN benchmark for fixed input sizes
    torch.backends.cudnn.benchmark = True
    
    if not hasattr(args, 'gpu'):
        args.gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.output_dir:
        # Add timestamp and config details to output dir name for clarity
        if utils.is_main_process():
            run_name = f"T{args.T}_tau{args.tau}_alpha{args.alpha}_lr{args.lr}_bs{args.batch_size}"
            args.output_dir = os.path.join(args.output_dir, run_name)
            os.makedirs(args.output_dir, exist_ok=True)
            
    device = torch.device(args.gpu)

    # Data loaders
    print(f"Loading data from {args.data_path}")
    if args.dataset == 'cifar10':
        num_classes = 10
        train_loader, test_loader, train_sampler = get_cifar10_loaders(
            args.data_path, args.batch_size, args.workers, args.distributed)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_loader, test_loader, train_sampler = get_imagenet_loaders(
            args.data_path, args.batch_size, args.workers, args.distributed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Model
    print(f"Creating model: {args.model}")
    neuron_params = {'tau': args.tau, 'surrogate_alpha': args.alpha, 'detach_reset': True}
    
    if args.model == 'resnet19':
        model = ResNet19(T=args.T, num_classes=num_classes, neuron_params=neuron_params)
    elif args.model == 'resnet18':
        model = ResNet18(T=args.T, num_classes=num_classes, neuron_params=neuron_params)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)

    # 2. PyTorch 2.0 Compile
    # SNNs benefit significantly from kernel fusion of element-wise LIF operations.
    # We try to compile the model if PyTorch 2.0+ is available.
    # Disabled due to InductorError related to missing libcuda.so
    # try:
    #     # 'reduce-overhead' is good for small batches/models, 'max-autotune' is more aggressive.
    #     # default mode is usually safest to start.
    #     model = torch.compile(model)
    #     print("PyTorch 2.0 compilation enabled.")
    # except Exception:
    #     pass

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Optimizer & Scaler
    # 3. Use SGD with Momentum (Better generalization than Adam)
    print(f"Using SGD: lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning Rate Scheduler with Warmup
    # Standard Cosine Annealing
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5)
    
    # Linear Warmup for first 5 epochs
    warmup_epochs = 5
    if args.epochs > warmup_epochs:
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.amp.GradScaler(device=args.gpu) # For mixed precision

    # Monitor & Writer
    writer = None
    monitor = None
    if utils.is_main_process() and args.output_dir:
        writer = SummaryWriter(log_dir=args.output_dir)
        monitor = SNNMonitor(writer)

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.start_epoch >= args.epochs:
            args.start_epoch = 0 # Restart if finished

    if args.test_only:
        evaluate(model, test_loader, device, args.T)
        return

    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        train_one_epoch(model, optimizer, train_loader, device, epoch, scaler, writer, args)
        lr_scheduler.step()
        
        # Validate & Monitor
        # Use monitor only on main process and only for a subset of test data to save time/space
        acc = evaluate(model, test_loader, device, epoch, writer, monitor, args)
        
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

    total_time = time.time() - start_time
    print(f"Training time {total_time}")

def train_one_epoch(model, optimizer, loader, device, epoch, scaler, writer, args):
    model.train()
    # Disable monitor mode for training
    if hasattr(model, 'module'):
        # access underlying model if DDP
        pass # We rely on default being False
    
    metric_logger = AverageMeter('Loss', ':.4f')
    acc_logger = AverageMeter('Acc', ':.2f')
    
    header = f'Epoch: [{epoch}]'
    
    for i, (image, target) in enumerate(loader):
        image, target = image.to(device), target.to(device)
        
        # Reset SNN state
        if hasattr(model, 'reset'):
            model.reset()
        elif hasattr(model, 'module') and hasattr(model.module, 'reset'):
            model.module.reset()
            
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=args.gpu):
            output = model(image)
            loss = nn.CrossEntropyLoss()(output, target)
            
        scaler.scale(loss).backward()
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        acc1, = accuracy(output, target, topk=(1,))
        batch_size = image.shape[0]
        metric_logger.update(loss.item(), batch_size)
        acc_logger.update(acc1.item(), batch_size)
        
        if i % 50 == 0 and utils.is_main_process():
            lr = optimizer.param_groups[0]["lr"]
            print(f"{header} [{i}/{len(loader)}] lr: {lr:.6f}  {str(metric_logger)}  {str(acc_logger)}")
            if writer:
                global_step = epoch * len(loader) + i
                writer.add_scalar('Train/Loss', metric_logger.avg, global_step)
                writer.add_scalar('Train/Acc', acc_logger.avg, global_step)

def evaluate(model, loader, device, epoch=0, writer=None, monitor=None, args=None):
    model.eval()
    loss_logger = AverageMeter('Loss', ':.4f')
    acc_logger = AverageMeter('Acc', ':.2f')
    
    # Enable monitor for the first batch only to save IO/Time
    # Or a few batches.
    monitor_enabled_once = False
    
    with torch.no_grad():
        for i, (image, target) in enumerate(loader):
            image, target = image.to(device), target.to(device)
            
            # Reset SNN
            if hasattr(model, 'reset'):
                model.reset()
            elif hasattr(model, 'module') and hasattr(model.module, 'reset'):
                model.module.reset()
            
            # Hook handling: Register -> Forward -> Remove
            # We only monitor the first batch of evaluation
            if monitor and not monitor_enabled_once and i == 0:
                monitor.register(model)
                monitor.set_monitor_mode(model, True)
                monitor_enabled_once = True
            
            output = model(image)
            loss = nn.CrossEntropyLoss()(output, target)
            
            acc1, = accuracy(output, target, topk=(1,))
            
            batch_size = image.shape[0]
            loss_logger.update(loss.item(), batch_size)
            acc_logger.update(acc1.item(), batch_size)
            
            if monitor and monitor_enabled_once and i == 0:
                monitor.set_monitor_mode(model, False)
                monitor.flush(epoch)
                monitor.remove() # Remove hooks immediately

    print(f" * Acc@1 {acc_logger.avg:.3f} Loss {loss_logger.avg:.4f}")
    
    if writer:
        writer.add_scalar('Test/Acc', acc_logger.avg, epoch)
        writer.add_scalar('Test/Loss', loss_logger.avg, epoch)
        
    return acc_logger.avg

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
