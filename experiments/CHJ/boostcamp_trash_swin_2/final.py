_base_ = [
    'cascade_rcnn_r50_fpn.py',
    'dataset.py',
    'schedule_1x.py',
    'default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
        neck=dict(in_channels=[96, 192, 384, 768]))
    # [128,256,512,1024] neck 교환 가능

evaluation = dict(
    interval = 1,
    metric = 'bbox',
    save_best='bbox_mAP_50',
    classwise =True
)