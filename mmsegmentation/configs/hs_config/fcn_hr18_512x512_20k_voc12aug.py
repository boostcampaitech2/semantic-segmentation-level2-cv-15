_base_ = [
    './fcn_hr18.py', './segmentation_data.py',
    './default_runtime.py', './schedule_20k.py'
]
model = dict(decode_head=dict(num_classes=11))