import numpy as np
from numpy import random
import mmcv
import albumentations as A

from ..builder import PIPELINES

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
        if self.bg_prop_limit < get_bg_proportion:

            # 아닌 경우 CropNonEmptyMaskIfExists 적용
            transform = A.CropNonEmptyMaskIfExists(height=self.height, width=self.width, 
            ignore_values=self.ignore_values, ignore_channels=self.ignore_channels, 
            always_apply=self.always_apply, p=self.p)

            transform = transform(image=img, mask=mask)

            results['img'] = transform['image']
            results['gt_semantic_seg'] = transform['mask']

        return results