# 참조 : deeplabv3plus_s101-d8_512x512_160k_ade20k.py

_base_ = [
    './models/deeplabv3plus_r50-d8.py', './datasets/ade20k.py',
    './runtime/default_runtime.py', './schedules/schedule_160k.py'
]

model = dict(
    pretrained='open-mmlab://resnest101',
    
    backbone=dict(
        type='ResNeSt',
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-5,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)