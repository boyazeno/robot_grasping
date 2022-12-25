from abc import ABC, abstractmethod
import torch
import torchvision
from torchvision.models.efficientnet import EfficientNet
import torch.nn as nn

class ModelBase(ABC):
    def __init__(self) -> None:
        super().__init__()


def get_efficientnet_b1(in_channels:int, out_channels:int)->EfficientNet:
    en = torchvision.models.efficientnet_b1(weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2)
    en.features[0][0] = type(en.features[0][0])(in_channels, en.features[0][0].out_channels, kernel_size=3, stride=2, padding=en.features[0][0].padding, dilation=en.features[0][0].dilation, groups=en.features[0][0].groups, bias=en.features[0]
    [0].bias)

    en.classifier[1] = nn.Linear(en.classifier[1].in_features, out_channels)
    return en


class SimpleQNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._state_backbone = nn.Sequential(*[get_efficientnet_b1(in_channels=8, out_channels=500), nn.ReLU()])
        layers = [nn.Linear(in_features=8*11, out_features=500), nn.ReLU()]
        self._action_backbone = nn.Sequential(*layers)
        self._output_layer = nn.Sequential(*[nn.Linear(in_features=1000, out_features=1)])

    def forward(self, s:torch.Tensor, a:torch.Tensor)->torch.Tensor:
        s_feature = self._state_backbone(s)
        a_feature = self._action_backbone(a)
        feature = torch.concat([s_feature, a_feature], dim=1)
        scores = self._output_layer(feature)
        return scores


class SimplePolicyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._state_backbone = get_efficientnet_b1(in_channels=8, out_channels=8*11)
    
    def forward(self, s:torch.Tensor)->torch.Tensor:
        feature = self._state_backbone(s)
        feature = feature.reshape(-1, 8, 11)
        feature = torch.softmax(feature, dim=-1)
        return feature.reshape(-1, 8*11)