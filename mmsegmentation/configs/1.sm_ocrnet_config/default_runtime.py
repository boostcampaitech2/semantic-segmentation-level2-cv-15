dataset_fold_num = 2

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),

        dict(type='WandbLoggerHook', interval=100,
            init_kwargs=dict(
                project = 'test',
                entity = 'ptop',
                name = f'hrnet_ocr(plain)_fold_{dataset_fold_num}'
            ),
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
#cudnn_benchmark = True
