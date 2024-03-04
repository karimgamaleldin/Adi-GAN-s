import torch
from torch import nn, optim
import pytorch_lightning as pl
from discriminator import Discriminator
from generator import Generator
import torchvision
from config import LR, BATCH_SIZE, BETA1, BETA2, L1_LAMBDA

class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.BCE = nn.BCEWithLogitsLoss()
        self.L1 = nn.L1Loss()
        self.automatic_optimization = False
  
    def forward(self, z):
        '''
        Forward pass through the generator
        '''
        return self.generator(z)
  
    def training_step(self, batch, batch_idx):
        sketch, img = batch['sketch'], batch['image'] 
        g_optim, d_optim = self.optimizers()
        
        # train discriminator
        y_fake = self.generator(sketch)
        D_real = self.discriminator(sketch, img)
        D_fake = self.discriminator(sketch, y_fake.detach())
        D_real_loss = self.BCE(D_real, torch.ones_like(D_real)) # y log (y_hat)
        D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake)) # (1 - y) log(1 - y_hat)
        D_loss = (D_real_loss + D_fake_loss) / 2 # according to the paper we divide by 2 to make the discrimnator train slower
        self.log("d_loss", D_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        d_optim.zero_grad()
        self.manual_backward(D_loss)
        d_optim.step()
            
        # train generator
        y_fake = self.generator(sketch)
        D_fake = self.discriminator(sketch, img)
        G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake)) # maximize log D(G(Z)) to avoid vanishing gradients
        l1 = self.L1(y_fake, img) * L1_LAMBDA
        G_loss = G_fake_loss + l1
        self.log("g_loss", G_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        g_optim.zero_grad()
        self.manual_backward(G_loss)
        g_optim.step()
        

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            sketch, img = batch['sketch'], batch['image']
            generated_img = self.generator(sketch)
            self.log_images(img, generated_img, 'val_examples')
        

    def configure_optimizers(self):
        g_optim = optim.Adam(self.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
        d_optim = optim.Adam(self.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
        return g_optim, d_optim
    
    def log_images(self, imgs, gens, name):
        concat = torch.cat([imgs, gens], dim=0)
        grid = torchvision.utils.make_grid(concat, nrow=BATCH_SIZE)
        self.logger.experiment.add_image(name, grid, self.current_epoch)
  