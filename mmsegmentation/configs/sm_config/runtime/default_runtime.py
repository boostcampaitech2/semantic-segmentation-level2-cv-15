# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook', interval=1, by_epoch=False,
        init_kwargs=dict(
            project='test',
            entity='ptop',
            name='mmseg_deeplabv3_r101'
        ),
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val',1)]