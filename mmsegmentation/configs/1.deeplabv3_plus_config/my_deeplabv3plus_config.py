_base_ = './deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
#model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

model = dict(pretrained='/opt/ml/semantic-segmentation-level2-cv-15/mmsegmentation/pretrain/deeplab3_best_mIoU_epoch_40.pth', backbone=dict(depth=101))
load_from = '/opt/ml/semantic-segmentation-level2-cv-15/mmsegmentation/pretrain/deeplab3_best_mIoU_epoch_40.pth'


#model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
#load_from = '/opt/ml/semantic-segmentation-level2-cv-15/mmsegmentation/pretrain/deeplab3_best_mIoU_epoch_40.pth'