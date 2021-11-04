import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
                            ToTensorV2()
                            ])

val_transform = A.Compose([
                          ToTensorV2()
                          ])

test_transform = A.Compose([
                           ToTensorV2()
                           ])

train_transform1 = A.Compose([
                            # A.RandomScale(always_apply=False, p=0.2, interpolation=2, scale_limit=(-0.2, 0.01)),
                            A.HorizontalFlip(always_apply=False, p=0.5),
                            A.GaussNoise(always_apply=False, p=0.1, var_limit=(2.0, 5.0)),

                            A.CoarseDropout(always_apply=False, p=0.1, max_holes=2, max_height=2, max_width=2, min_holes=2, min_height=2, min_width=2),
                            A.OneOf([
                                A.RandomGamma(always_apply=False, p=1.0, gamma_limit=(110, 140), eps=1e-07),
                                A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(0, 0.03), contrast_limit=(-0.1, 0.1), brightness_by_max=True),
                            ], p=0.1),
                            A.OneOf([
                                A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-5, 5), sat_shift_limit=(-10, 10), val_shift_limit=(-5, 5)),
                                A.RGBShift(always_apply=False, p=1.0, r_shift_limit=(-5, 5), g_shift_limit=(-5, 5), b_shift_limit=(-5, 5)),
                            ], p=0.1),
                            ToTensorV2()
                            ])

def get_transform(transform_name):
    if transform_name == 'train_transform':
        return train_transform
    elif transform_name == 'val_transform':
        return val_transform
    elif transform_name == 'test_transform':
        return test_transform
    elif transform_name == 'train_transform1':
        return train_transform1