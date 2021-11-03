import albumentations as A
from ..builder import PIPELINES
import numpy as np
import cv2

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
        print("리절트ㅡ",results)
        img = results['img']

        transform = A.RandomBrightnessContrast(self.brightness_limit, self.contrast_limit, self.brightness_by_max, self.always_apply, self.p)
        transform = transform(image=img)
        results['img'] = transform['image']

        return results


@PIPELINES.register_module()
class CropNonEmptyMaskIfExists4(object):

    def __init__(self, 
                height=400, 
                width=400, 
                ignore_values=None, 
                ignore_channels=None, 
                always_apply=False,
                bg_prop_limit=0.5, p=1.0):

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
        if self.bg_prop_limit < get_bg_proportion:

            # 아닌 경우 CropNonEmptyMaskIfExists 적용
            transform = A.CropNonEmptyMaskIfExists(height=self.height, width=self.width, 
            ignore_values=self.ignore_values, ignore_channels=self.ignore_channels, 
            always_apply=self.always_apply, p=self.p)

            transform = transform(image=img, mask=mask)

            results['img'] = transform['image']
            results['gt_semantic_seg'] = transform['mask']

        return results


@PIPELINES.register_module()
class BoundingBoxesAugmentation(object):

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
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bboxes = results['bbox']
        print("리절트ㅡ",results)

        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2)
        ], bbox_params=A.BboxParams(format='coco')) 
        # also support min area / min visivility 0~1
        # ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))
        
        transformed = transform(image=image, bboxes=bboxes)
        results['img'] = transformed['image']
        results['bboxes'] = transformed['bboxes']

        return results
