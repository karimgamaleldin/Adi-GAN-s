from torch.utils.data import Dataset
import os
import cv2
from config import both_transform

class AdiGansDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_paths = os.listdir(self.data_dir)
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        p = self.data_paths[idx]
        path = os.path.join(self.data_dir, p)
        img = self.get_image(path)
        sketch, image = self.divide_sketch_image(img)
        
        augmentations = both_transform(image=image, sketch=sketch)
        sketch, image = augmentations['sketch'], augmentations['image']
        return {
            'sketch': sketch,
            'image': image
        }
    
    def divide_sketch_image(self, img):
        sketch = img[:, :len(img), :]
        image = img[:, len(img):, :]
        return sketch, image
    
    def get_image(self, path: str):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img