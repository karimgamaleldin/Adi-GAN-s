import torch
from torch import nn 
from blocks import DiscCNNBlock

class Discriminator(nn.Module):
  def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
    super().__init__()
    # initial layer
    self.initial = nn.Sequential(
      nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'), 
      nn.LeakyReLU(0.2)  
    )
    # making the rest layers except the final
    layers = []
    in_channels = features[0]
    for feature in features[1:]:
      layers.append(DiscCNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2))
      in_channels = feature
    # final layer
    layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))

    self.model = nn.Sequential(*layers)


  def forward(self, x, y):
    x = torch.cat([x, y], dim=1)
    x = self.initial(x)
    x = self.model(x)
    return x

def test():
  print("Testing Discriminator")
  x = torch.randn((1, 3, 256, 256))
  y = torch.randn((1, 3, 256, 256))
  model = Discriminator()
  preds = model(x, y)
  print(preds.shape)

test()
