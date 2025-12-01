import torch
import torch.nn as nn
from modules.lif_triton import LIFNodeTriton

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class SNNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, neuron_params=None):
        super().__init__()
        self.neuron_params = neuron_params or {}
        
        # Standard Layers (Stateless over T)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = LIFNodeTriton(**self.neuron_params)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.lif2 = LIFNodeTriton(**self.neuron_params)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # x is (N*T, C, H, W)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.lif2(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, T=4, num_classes=1000, neuron_params=None):
        super().__init__()
        self.T = T
        self.neuron_params = neuron_params or {'tau': 2.0, 'detach_reset': True}
        
        self.inplanes = 64
        
        # Standard Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = LIFNodeTriton(**self.neuron_params)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SNNBasicBlock.expansion, num_classes)
        
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
                conv1x1(self.inplanes, planes * SNNBasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * SNNBasicBlock.expansion),
            )

        layers = []
        layers.append(SNNBasicBlock(self.inplanes, planes, stride, downsample, self.neuron_params))
        self.inplanes = planes * SNNBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(SNNBasicBlock(self.inplanes, planes, neuron_params=self.neuron_params))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (N, 3, H, W)
        
        # 1. Static Encoding
        x = self.conv1(x)
        x = self.bn1(x)
        # x: (N, 64, H/2, W/2)
        
        # 2. Global Flattening (N -> N*T)
        # (N, C, H, W) -> (N, T, C, H, W) -> (N*T, C, H, W)
        x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1).flatten(0, 1)
        
        # 3. Network (Processing N*T batch)
        x = self.lif1(x)      # Handles internal reshape
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # (N*T, 512)
        x = self.fc(x)          # (N*T, NumClasses)
        
        # 4. Aggregate Time
        # (N*T, NumClasses) -> (N, T, NumClasses) -> Mean -> (N, NumClasses)
        # Note: We need to know N. N = Batch // T
        batch_size = x.shape[0]
        N = batch_size // self.T
        x = x.view(N, self.T, -1)
        return torch.mean(x, dim=1)