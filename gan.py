import torch
from torch import nn, optim
import pytorch_lightning as pl
from discriminator import Discriminator
from generator import Generator

class GAN(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.generator = Generator()
    self.discriminator = Discriminator()
  
  def forward(self, z):
    '''
    Forward pass through the generator
    '''
    return self.generator(z)
  
  def training_step(self, batch, batch_idx, optimizer_idx):
    sketch, img = batch['sketch'], batch['image'] 

    # sample noise
    z = torch.randn(img.size(0), 3, 256, 256).to(self.device)

    # train generator
    if optimizer_idx == 0:
      loss = 0
      self.log("g_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      pass 

    # train discriminator
    if optimizer_idx == 1:
      loss = 0
      self.log("d_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      pass
    return loss

  def validation_step(self, batch, batch_idx):
    sketch, img = batch['sketch'], batch['image']
    z = torch.randn(img.size(0), 3, 256, 256).to(self.device)
    generated_img = self.generator(z)
    real_pred = self.discriminator(sketch, img)
    fake_pred = self.discriminator(sketch, generated_img)
    return real_pred, fake_pred

  def configure_optimizers(self):
    g_optim = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optim = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return [g_optim, d_optim], []