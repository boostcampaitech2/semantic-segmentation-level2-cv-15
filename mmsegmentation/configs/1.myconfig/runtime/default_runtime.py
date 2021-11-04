dataset_fold_num = 4

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',interval=100,
            init_kwargs=dict(
                project='test',
                entity = 'ptop',
                name = f'swin(second_pseudo)_fold_{dataset_fold_num}'
            ),
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
