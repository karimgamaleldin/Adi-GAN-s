from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
import cv2
import numpy as np
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
    
    def setup(self):
        sketch_train, images_train = self.get_data()
        self.train_dataset = AdiGansDataset(sketch_train, images_train)
        sketch_val, images_val = self.get_data(train=False)
        self.val_dataset = AdiGansDataset(sketch_val, images_val)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def divide_sketch_image(self, img):
        sketch = img[:, :len(img), :]
        image = img[:, len(img):, :]
        return sketch, image
    
    def get_image(self, path: str):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_data(self, train=True):
        print('getting data')
        dir_used = self.train_dir if train else self.val_dir
        paths = os.listdir(dir_used)
        sketches = []
        images = []
        for i, p in enumerate(paths[:1000]):
            path = os.path.join(dir_used, p)
            img = self.get_image(path)
            sketch, img = self.divide_sketch_image(img)
            sketches.append(sketch)
            images.append(img)
            if i % 1000 == 0:
                print(i)

        sketches = np.array(sketches, dtype=np.float32)
        images = np.array(images, dtype=np.float32)
        return sketches, images
        