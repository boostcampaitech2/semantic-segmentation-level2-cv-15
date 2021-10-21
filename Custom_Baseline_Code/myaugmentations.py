import albumentations as A
from albumentations.augmentations.geometric.resize import RandomScale
from albumentations.augmentations.transforms import RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    RandomScale(p=0.5, interpolation=0, scale_limit=(-0.5, 0.5)),
    RandomBrightnessContrast(p=0.5, brightness_limit=(0.0, 0.2), contrast_limit=(0.0, 0.2), brightness_by_max=True),
    ToTensorV2()
    ])

train_transform_1 = A.Compose([
    ToTensorV2()
    ])

val_transform = A.Compose([
                          ToTensorV2()
                          ])

test_transform = A.Compose([
                           ToTensorV2()
                           ])