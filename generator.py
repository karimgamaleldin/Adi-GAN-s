import torch
from torch import nn, optim 
from torch.nn import functional as F
import pytorch_lightning as pl
from blocks import GenCNNBlock

class Generator(pl.LightningModule):
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

    self.up1 = GenCNNBlock(features*8, features*8, down=False, act="relu", use_dropout=True)
    self.up2 = GenCNNBlock(features*16, features*8, down=False, act="relu", use_dropout=True)
    self.up3 = GenCNNBlock(features*16, features*8, down=False, act="relu", use_dropout=True)
    self.up4 = GenCNNBlock(features*16, features*8, down=False, act="relu", use_dropout=False)
    self.up5 = GenCNNBlock(features*16, features*4, down=False, act="relu", use_dropout=False)
    self.up6 = GenCNNBlock(features*8, features*2, down=False, act="relu", use_dropout=False)
    self.up7 = GenCNNBlock(features*4, features, down=False, act="relu", use_dropout=False)

    self.final = nn.Sequential(
      nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
      nn.Tanh()
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
    u2 = self.up2(torch.cat([u1, d7], dim=1))
    u3 = self.up3(torch.cat([u2, d6], dim=1))
    u4 = self.up4(torch.cat([u3, d5], dim=1))
    u5 = self.up5(torch.cat([u4, d4], dim=1))
    u6 = self.up6(torch.cat([u5, d3], dim=1))
    u7 = self.up7(torch.cat([u6, d2], dim=1))
    u8 = self.final(torch.cat([u7, d1], dim=1))
    return u8
  
  
  def training_step(self, batch, batch_idx):
    pass 

  def validation_step(self, batch, batch_idx):
    pass

  def test_step(self, batch, batch_idx):
    pass 

  def predict_step(self, batch, batch_idx, dataloader_idx=None):
    pass

  def configure_optimizers(self):
    pass


def test():
  print("Testing Generator")
  x = torch.randn((1, 3, 256, 256))
  model = Generator()
  preds = model(x)
  print(preds.shape)

test()