_base_ = [
    '../_base_/models/fcn_hr18.py', './Dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(decode_head=dict(num_classes=11))
