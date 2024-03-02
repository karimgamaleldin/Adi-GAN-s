import torch
from torch import nn


class DiscCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)
    

class GenCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        # Getting conv2d or conv2dtranspose
        cnn = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        # Getting relu or leakyrelu
        act = nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2)
        # Layers
        self.conv = nn.Sequential(
            cnn,
            nn.BatchNorm2d(out_channels),
            act
        )
        # Adding dropout
        if use_dropout:
            self.conv.add_module("dropout", nn.Dropout(0.5))


    def forward(self, x):
        return self.conv(x)