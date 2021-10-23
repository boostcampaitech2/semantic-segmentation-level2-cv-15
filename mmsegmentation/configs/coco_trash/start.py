_base_ = [
    './deeplabv3_r50-d8.py', './cityscapes.py',
    './default_runtime.py', './schedule_80k.py'
]

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=11,
    ),
    auxiliary_head=dict(in_channels=256, channels=11))
