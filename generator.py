import torch
from torch import nn 
from blocks import GenCNNBlock

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.initial = nn.Sequential(
          nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
          nn.LeakyReLU(0.2, inplace=True)
        )

        self.down1 = GenCNNBlock(features, features*2, down=True, act="LeakyReLU", use_dropout=False)
        self.down2 = GenCNNBlock(features*2, features*4, down=True, act="LeakyReLU", use_dropout=False)
        self.down3 = GenCNNBlock(features*4, features*8, down=True, act="LeakyReLU", use_dropout=False)
        self.down4 = GenCNNBlock(features*8, features*8, down=True, act="LeakyReLU", use_dropout=False)
        self.down5 = GenCNNBlock(features*8, features*8, down=True, act="LeakyReLU", use_dropout=False)
        self.down6 = GenCNNBlock(features*8, features*8, down=True, act="LeakyReLU", use_dropout=False)

        self.bottleneck = nn.Sequential(
          nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"),
          nn.ReLU(inplace=True)
        )

        self.up1 = GenCNNBlock(features*8, features*8, down=False, act="relu", use_dropout=True) # 8 channels from bottleneck
        self.up2 = GenCNNBlock(features*16, features*8, down=False, act="relu", use_dropout=True) # 8 channels from up1 and 8 channels from down7(skip connection)
        self.up3 = GenCNNBlock(features*16, features*8, down=False, act="relu", use_dropout=True) # 8 channels from up2 and 8 channels from down6(skip connection)
        self.up4 = GenCNNBlock(features*16, features*8, down=False, act="relu", use_dropout=False) # 8 channels from up3 and 8 channels from down5(skip connection)
        self.up5 = GenCNNBlock(features*16, features*4, down=False, act="relu", use_dropout=False) # 8 channels from up4 and 8 channels from down4(skip connection)
        self.up6 = GenCNNBlock(features*8, features*2, down=False, act="relu", use_dropout=False) # 4 channels from up5 and 4 channels from down3(skip connection)
        self.up7 = GenCNNBlock(features*4, features, down=False, act="relu", use_dropout=False) # 2 channels from up6 and 2 channels from down2(skip connection)

        self.final = nn.Sequential(
          nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1), # 2 channels from up7 and 2 channels from down1(skip connection)
          nn.Tanh() # tanh activation function to scale the pixel values between -1 and 1
        )

    def forward(self, x):
        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1)) # Concatenating u1 and d7 along the channels dimension
        u3 = self.up3(torch.cat([u2, d6], dim=1)) # Concatenating u2 and d6 along the channels dimension
        u4 = self.up4(torch.cat([u3, d5], dim=1)) # Concatenating u3 and d5 along the channels dimension
        u5 = self.up5(torch.cat([u4, d4], dim=1)) # Concatenating u4 and d4 along the channels dimension
        u6 = self.up6(torch.cat([u5, d3], dim=1)) # Concatenating u5 and d3 along the channels dimension
        u7 = self.up7(torch.cat([u6, d2], dim=1)) # Concatenating u6 and d2 along the channels dimension
        u8 = self.final(torch.cat([u7, d1], dim=1)) # Concatenating u7 and d1 along the channels dimension
        return u8
  


def test():
  print("Testing Generator")
  x = torch.randn((1, 3, 256, 256))
  model = Generator()
  preds = model(x)
  print(preds.shape)

test()