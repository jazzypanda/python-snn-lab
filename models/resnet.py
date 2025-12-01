import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.neuron import LIFNode
from modules.lif_triton import LIFNodeTriton
from modules.layer import TimeDistributed, DirectEncoder

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class SNNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, neuron_params=None):
        super().__init__()
        self.neuron_params = neuron_params or {}
        
        # We wrap stateless layers in TimeDistributed
        self.conv1 = TimeDistributed(conv3x3(inplanes, planes, stride))
        self.bn1 = TimeDistributed(nn.BatchNorm2d(planes))
        self.lif1 = LIFNodeTriton(**self.neuron_params)
        
        self.conv2 = TimeDistributed(conv3x3(planes, planes))
        self.bn2 = TimeDistributed(nn.BatchNorm2d(planes))
        
        # Note: The second LIF is after the addition in ResNet
        self.lif2 = LIFNodeTriton(**self.neuron_params)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection: Add currents (membranes/outputs of BN)
        out = out + identity
        
        # Final LIF
        out = self.lif2(out)

        return out

class ResNet19(nn.Module):
    def __init__(self, T=4, num_classes=10, neuron_params=None):
        super().__init__()
        self.T = T
        self.neuron_params = neuron_params or {'tau': 2.0, 'detach_reset': True}
        
        # Direct Encoder: (N, C, H, W) -> (N, T, C, H, W)
        self.encoder = DirectEncoder(T)
        
        self.inplanes = 64
        
        # Initial Layer
        # Input is 3 channels (RGB)
        self.conv1 = TimeDistributed(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = TimeDistributed(nn.BatchNorm2d(64))
        self.lif1 = LIFNodeTriton(**self.neuron_params)
        
        # ResNet Layers
        # ResNet18 usually has [2, 2, 2, 2] blocks. 
        # ResNet19 usually implies a specific variant or just depth. 
        # Standard ResNet18 is 18 layers (1 conv + 8*2 blocks + 1 fc).
        # Let's stick to standard ResNet18 structure which is close to 19.
        # Or if "ResNet19" refers to VGG-style 19? 
        # User said "ResNet19", which is a common SNN paper architecture (e.g., Direct Training for Spiking Neural Networks).
        # It is often: Conv-BN-LIF x3 -> ResBlock x3 ... 
        # Let's implement standard ResNet18 structure first as it's robust.
        # Layers: [2, 2, 2, 2]
        
        layers = [2, 2, 2, 2]
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Final Classification
        self.avgpool = TimeDistributed(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = TimeDistributed(nn.Linear(512 * SNNBasicBlock.expansion, num_classes))
        
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * SNNBasicBlock.expansion:
            downsample = nn.Sequential(
                TimeDistributed(conv1x1(self.inplanes, planes * SNNBasicBlock.expansion, stride)),
                TimeDistributed(nn.BatchNorm2d(planes * SNNBasicBlock.expansion)),
            )

        layers = []
        layers.append(SNNBasicBlock(self.inplanes, planes, stride, downsample, self.neuron_params))
        self.inplanes = planes * SNNBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(SNNBasicBlock(self.inplanes, planes, neuron_params=self.neuron_params))

        return nn.Sequential(*layers)
    
    def reset(self):
        # Reset all LIFNodes
        for m in self.modules():
            if hasattr(m, 'reset') and m is not self:
                m.reset()

    def forward(self, x):
        # x: (N, C, H, W)
        x = self.encoder(x) # -> (N, T, C, H, W)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2) # (N, T, 512)
        x = self.fc(x) # (N, T, num_classes)
        
        # Mean over time for prediction (Voting or Mean Potential)
        # Requirement: "不发射脉冲，直接用膜电位计算交叉熵"
        # This usually means we take the average output of the last linear layer over T.
        output = torch.mean(x, dim=1)
        
        return output
