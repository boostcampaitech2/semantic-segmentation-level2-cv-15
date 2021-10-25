import torch.nn as nn
from torchvision import models
from deeplab_xception_model.deeplab import *

def config_model(model_name, num_classes):
    model = None
    available_models = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3+']

    if model_name in available_models:
        try:
            if model_name == 'fcn_resnet50':
                model = models.segmentation.fcn_resnet50(pretrained=True)
                model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
            elif model_name == 'fcn_resnet101':
                model = models.segmentation.fcn_resnet101(pretrained=True)
                model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
            elif model_name == 'deeplabv3+':
                model = DeepLab(backbone='resnet', num_classes=num_classes) # backbone=xception 오류 발생.
            
        except Exception:
            raise ValueError(f"The model({model_name}) occurs error during revising classifier.")
    
    else:
        raise ValueError(f"The model({model_name}) is not available.\nYou can use only these models({available_models})")
    
    return model