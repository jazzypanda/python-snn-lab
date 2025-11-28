# Python SNN Lab

This repository contains a PyTorch implementation for training Spiking Neural Networks (SNN) on the CIFAR-10 dataset. It features a custom `ResNet19` SNN architecture and utilizes surrogate gradients for training.

## Project Structure

- `train.py`: Main training script.
- `models/`: Contains model definitions (e.g., `ResNet19`).
- `modules/`: Custom SNN modules (Neurons, Layers, Surrogate gradients).
- `utils/`: Utility scripts for metrics, monitoring, and distributed training.
- `data/`: Data loading scripts.
- `configs/`: Configuration files (if applicable).

## Requirements

- Python 3.8+
- PyTorch
- TensorBoard

## Usage

### Training

To start training the ResNet19 model on CIFAR-10:

```bash
python train.py --model resnet19 --epochs 100 --batch-size 64 --lr 1e-3
```

### Distributed Training

To run with multiple GPUs (e.g., 2 GPUs):

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py --distributed --batch-size 64
```

## Features

- **LIF Neuron**: Leaky Integrate-and-Fire neuron model with surrogate gradient support.
- **TimeDistributed Layers**: Wrappers to handle temporal dimensions in SNNs.
- **ResNet19 Architecture**: Adapted for SNNs with direct encoding.
- **Monitoring**: TensorBoard support for tracking loss, accuracy, and other metrics.

## License

[MIT](LICENSE)
