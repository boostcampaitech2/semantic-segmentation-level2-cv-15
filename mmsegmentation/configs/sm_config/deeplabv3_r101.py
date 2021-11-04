_base_ = [
    'dataset/coco-trash.py', 'runtime/default_runtime.py',
    'schedules/schedule_80k.py', 'models/deeplabv3_r101-d8_512x512_20k_voc12aug.py',
]