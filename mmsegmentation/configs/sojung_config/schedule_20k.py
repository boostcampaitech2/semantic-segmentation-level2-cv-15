# optimizer
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
#lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=True)
lr_config = dict(policy='poly', power=0.9, min_lr=0.00001, by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(by_epoch=True, interval=2, max_keep_ckpts=5)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, save_best='mIoU')
