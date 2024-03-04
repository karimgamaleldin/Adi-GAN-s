from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import AdiGansDataset
from config import BATCH_SIZE

class AdiGansDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=BATCH_SIZE):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
    
    def prepare_data(self):
        pass 
    
    def setup(self, stage=None):
        self.train_dataset = AdiGansDataset(self.train_dir)
        self.val_dataset = AdiGansDataset(self.val_dir)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=3)
        