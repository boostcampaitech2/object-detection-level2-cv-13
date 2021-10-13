# dataset settings
dataset_type = 'CocoDataset'
data_root = '../dataset/'
classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

           
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.01),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/trn_val_split_json/train_split_1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline), 
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/trn_val_split_json/valid_split_1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict( 
        type=dataset_type,
        ann_file=data_root + '/trn_val_split_json/valid_split_1.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')