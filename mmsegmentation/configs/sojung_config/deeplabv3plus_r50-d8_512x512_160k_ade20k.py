_base_ = [
    './models/deeplabv3plus_r50-d8.py',
    './custdata.py',
    './default_runtime.py',
    './schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))
