_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/segmentation_data.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(decode_head=dict(num_classes=11))
