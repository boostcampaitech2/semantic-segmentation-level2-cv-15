_base_ = './deeplabv3_r50-d8.py'
model = dict(
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))
