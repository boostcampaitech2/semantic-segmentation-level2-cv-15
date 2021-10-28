# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', interval=100,
        init_kwargs=dict(
            project='test',
            entity='ptop',
            name='mmseg_deeplabv3_r101_test'
        ),
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1),('val', 1)]
cudnn_benchmark = True
