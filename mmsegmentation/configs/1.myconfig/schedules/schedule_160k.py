# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)


# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=100000)
# checkpoint_config = dict(by_epoch=False, interval=100000)
# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)

runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(interval=500)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, save_best='mIoU')