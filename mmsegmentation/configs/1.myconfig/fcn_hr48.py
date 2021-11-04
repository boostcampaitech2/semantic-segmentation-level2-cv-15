# 참조 : fcn_hr48_512x1024_160k_cityscapes.py

_base_ = [
    './models/fcn_hr18.py', './datasets/cityscapes.py',
    './runtime/default_runtime.py', './schedules/schedule_160k.py'
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        num_classes=11
        )
    )