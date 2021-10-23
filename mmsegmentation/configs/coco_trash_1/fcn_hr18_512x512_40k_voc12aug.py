_base_ = [
    './fcn_hr18.py', './pascal_voc12_aug.py',
    './default_runtime.py', './schedule_40k.py'
]
model = dict(decode_head=dict(num_classes=11))
