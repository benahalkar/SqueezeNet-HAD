import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, input_ch: int, squeeze_ch: int, e1x1_ch: int, e3x3_ch: int) -> None:
        super(FireModule, self).__init__()
        self.squeeze       = nn.Conv2d(input_ch, squeeze_ch, kernel_size=1)
        self.squeeze_act   = nn.ReLU(inplace=True)

        self.expand1x1     = nn.Conv2d(squeeze_ch, e1x1_ch, kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)

        self.expand3x3     = nn.Conv2d(squeeze_ch, e3x3_ch, kernel_size=3, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.squeeze_act(self.squeeze(x))
        return torch.cat([
            self.expand1x1_act(self.expand1x1(x)), 
            self.expand3x3_act(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

def build_model(num_classes, weights_path, activation="relu", verbose=False):
    """
      https://github.com/vfinotti/flower-or-crops-cnn-classification-pytorch/blob/master/squeezenet-training.py
      
    """
    # initializing the model
    model = SqueezeNet()
    model.load_state_dict(torch.load(weights_path))
    model.classifier[1].out_channels = num_classes
    if verbose: print(model)

    for param in model.features.parameters():
        param.require_grad = False

    num_features = model.classifier[1].in_channels
    features = list(model.classifier.children())[:-3]                      # Remove last 3 layers
    features.extend([nn.Conv2d(num_features, num_classes, kernel_size=1)]) # Add layers individually
    
    if activation == "tanh": features.extend([nn.Tanh()])    
    elif activation == "leaky_relu": features.extend([nn.LeakyReLU(inplace=True)])  
    else: features.extend([nn.ReLU(inplace=True)])
    # features.extend([nn.ReLU(inplace=True)])
    features.extend([nn.AdaptiveAvgPool2d(output_size=(1,1))]) 
    model.classifier = nn.Sequential(*features)                            # Replace the model classifier
    return model