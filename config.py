import albumentations as A
from albumentations.pytorch import ToTensorV2

# constants and configurations
LR = 2e-4
BATCH_SIZE = 4
IMG_SIZE = 256
CHANNELS_IMG = 3
EPOCHS = 15
BETA1 = 0.5
BETA2 = 0.999
L1_LAMBDA = 100


both_transform = A.Compose(
    [A.HorizontalFlip(p=0.5),
     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value = 255.0),
     ToTensorV2()
    ],
    additional_targets={'sketch': 'image'}
)