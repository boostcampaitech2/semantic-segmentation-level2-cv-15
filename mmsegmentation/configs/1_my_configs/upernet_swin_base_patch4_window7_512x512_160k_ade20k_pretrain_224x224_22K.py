_base_ = [
    './upernet_swin_base_patch4_window7_512x512_160k_ade20k_'
    'pretrain_224x224_1K.py'
]
model = dict(pretrained='/opt/ml/semantic-segmentation-level2-cv-15/mmsegmentation/pretrain/best_mIoU_epoch_26.pth')
