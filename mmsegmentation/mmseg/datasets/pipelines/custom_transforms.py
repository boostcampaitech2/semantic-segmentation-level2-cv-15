from ..builder import PIPELINES
import albumentations as A
import numpy as np

@PIPELINES.register_module()
class RandomBrightnessContrast(object):

    def __init__(self, 
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                brightness_by_max=True, 
                always_apply=False, p=0.5):

        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_by_max = brightness_by_max
        self.always_apply = always_apply
        self.p = p
    

    def __call__(self, results):

        img = results['img']

        transform = A.RandomBrightnessContrast(self.brightness_limit, self.contrast_limit, self.brightness_by_max, self.always_apply, self.p)
        transform = transform(image=img)
        results['img'] = transform['image']

        return results

@PIPELINES.register_module()
class CropNonEmptyMaskIfExists(object):

    def __init__(self, 
                height=400, 
                width=400, 
                ignore_values=None, 
                ignore_channels=None, 
                always_apply=False,
                bg_prop_limit=0.7, p=1.0):

        self.height = height
        self.width = width
        self.ignore_values = ignore_values
        self.ignore_channels = ignore_channels
        self.always_apply = always_apply
        self.bg_prop_limit = bg_prop_limit
        self.p = p


    def __call__(self, results):

        img = results['img']
        mask = results['gt_semantic_seg']

        gt_classes, counts = np.unique(mask ,return_counts=True)
        gt_class_count_dict = dict(zip(gt_classes, counts))

        get_bg_proportion = gt_class_count_dict[0] / sum(gt_class_count_dict.values())

        # 일정 비율 이상인 경우 그냥 리턴
        if self.bg_prop_limit <= get_bg_proportion:

            # 아닌 경우 CropNonEmptyMaskIfExists 적용
            transform = A.CropNonEmptyMaskIfExists(height=self.height, width=self.width, 
            ignore_values=self.ignore_values, ignore_channels=self.ignore_channels, 
            always_apply=self.always_apply, p=self.p)

            transform = transform(image=img, mask=mask)

            results['img'] = transform['image']
            results['gt_semantic_seg'] = transform['mask']

        return results

@PIPELINES.register_module()
class GridDropout(object):

    def __init__(self,
                ratio=0.5, 
                unit_size_min=None, 
                unit_size_max=None, 
                holes_number_x=None, 
                holes_number_y=None, 
                shift_x=0, shift_y=0, 
                random_offset=False, 
                fill_value=0, 
                mask_fill_value=None, 
                always_apply=False, 
                with_mask=False, p=0.5):

        self.ratio = ratio
        self.unit_size_min = unit_size_min
        self.unit_size_max = unit_size_max
        self.holes_number_x = holes_number_x
        self.holes_number_y = holes_number_y
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.random_offset = random_offset
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        self.always_apply = always_apply
        self.with_mask = with_mask
        self.p = p

    
    def __call__(self, results):

        img = results['img']
        mask = results['gt_semantic_seg']

        if self.with_mask:
            self.mask_fill_value = 0

        transform = A.GridDropout(self.ratio, self.unit_size_min, self.unit_size_max, self.holes_number_x, 
                                self.holes_number_y, self.shift_x, self.shift_y, self.random_offset,
                                self.fill_value, self.mask_fill_value, self.always_apply, self.p)
        
        transform = transform(image=img, mask=mask)
        results['img'] = transform['image']
        results['gt_semantic_seg'] = transform['mask']

        return results


@PIPELINES.register_module()
class GaussNoise(object):

    def __init__(self, 
                var_limit=(10.0, 50.0), 
                mean=0, 
                per_channel=True, 
                always_apply=False, 
                p=0.5):

        self.var_limit = var_limit
        self.mean = mean
        self.per_channel = per_channel
        self.always_apply = always_apply
        self.p = p


    def __call__(self, results):

        img = results['img']

        transform = A.GaussNoise(self.var_limit, self.mean, self.per_channel, self.always_apply, self.p)
        transform = transform(image=img)

        results['img'] = transform['image']
        return results

@PIPELINES.register_module()
class ColorJitter(object):

    def __init__(self, 
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.2, 
                always_apply=False, 
                p=0.5):

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.always_apply = always_apply
        self.p = p

    def __call__(self, results):
        img = results['img']

        transform = A.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue, self.always_apply, self.p)
        transform = transform(image=img)

        results['img'] = transform['image']
        return results