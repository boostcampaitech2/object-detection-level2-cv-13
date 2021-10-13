checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=50),
        dict(type='WandbLoggerHook',interval=300,
            init_kwargs=dict(
                project='swin_transformer',
                entity = 'cval',
                name = 'vanila_swin_transformer_MaskRCNN_from34epoch',
                
            ),
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# 1 epoch에 train과 validation을 모두 하고 싶으면 workflow = [('train', 1), ('val', 1)]