import torch
from torch.utils.data import Dataset

class AdiGansDataset(Dataset):
    def __init__(self, sketches, images):
        self.sketches = torch.tensor(sketches).float().permute((0, 3, 1, 2))
        self.images = torch.tensor(images).float().permute((0, 3, 1, 2))
        
        
    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch = self.sketches[idx]
        image = self.images[idx]
        return {
            'sketch': sketch,
            'image': image
        }