# dataset settings
#dataset_type = 'MytrashDataset'
dataset_type = 'CustomDataset'
data_root = '/opt/ml/semantic-segmentation-level2-cv-15/input/mmseg'

CLASSES = ('Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
PALETTE = [[0,0,0], [192,0,128], [192,192,255], [0,128,64], [128,0,0], [172,224,64], [244,64,60], [192,128,64], [255,200,224], [243,246,244], [128,0,192]]

# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(mean=[117.324, 112.09, 106.66], std=[53.76, 52.95, 55.22], to_rgb=True)

# crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=[(256, 256), (384, 384), (512, 512), (768, 768)],
    #     img_ratios=[0.75, 1.0, 1.25],
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', keep_ratio=True),
    #         dict(type='RandomFlip', prob=0.5),
    #     ]),
    # dict(type='Resize', img_scale=[(384, 384), (512, 512), (768, 768)], ratio_range=(0.75, 1.25)),
    # dict(type='Resize', img_scale=(512, 512), ratio_range=(0.75, 1.25)),
    
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.75, 1.0, 1.25],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

Fold = 2

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{data_root}/images/train_{Fold}',
        ann_dir=f'{data_root}/annotations/train_{Fold}',
        classes=CLASSES,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{data_root}/images/val_{Fold}',
        ann_dir=f'{data_root}/annotations/val_{Fold}',
        classes=CLASSES,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{data_root}/images/test',
        ann_dir=f'{data_root}/annotations/test',
        classes=CLASSES,
        pipeline=test_pipeline))
