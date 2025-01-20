import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.relu_squeeze = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.relu_expand1x1 = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.relu_expand3x3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.relu_squeeze(x)
        out1x1 = self.expand1x1(x)
        out1x1 = self.relu_expand1x1(out1x1)
        out3x3 = self.expand3x3(x)
        out3x3 = self.relu_expand3x3(out3x3)
        return torch.cat([out1x1, out3x3], 1)

class SqueezeNet_Residual(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet_Residual, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire5 = Fire(256, 32, 128, 128)
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.pool8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire9 = Fire(512, 64, 256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.conv10 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.relu_conv10 = nn.ReLU(inplace=True)
        self.pool10 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        bypass_23 = x  
        x = self.fire4(x)
        x = self.pool4(x)
        bypass_45 = x  
        x = self.fire5(x)
        bypass_67 = x  
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.pool8(x)
        bypass_89 = x  
        x = self.fire9(x)
        x = self.dropout(x)
        x = self.conv10(x)
        x = self.relu_conv10(x)
        x = self.pool10(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SqueezeNet_Residual(num_classes=10).to(device)