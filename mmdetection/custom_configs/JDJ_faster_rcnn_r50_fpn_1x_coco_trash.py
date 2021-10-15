_base_ = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_512'

########################################################################
seed = 2021
gpu_ids = [0]

# 데이터 경로
root='../../dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# wandb 연결
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='JDJ_exp',
                name='faster_rcnn_r50_fpn_1x_512',
                entity='cval'
            )
        )
    ])

# epoch 설정
runner = dict(type='EpochBasedRunner', max_epochs=50)

# validation metric 설정
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='bbox_mAP_50',
    classwise=True
)

# checkpoint
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
########################################################################

# model class 개수 설정
model = dict(
    roi_head = dict(
        bbox_head = dict(
            num_classes = 10
        )
    )
)

# data pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512,512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
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
    train = dict(
        classes = classes,
        img_prefix = root,
        ann_file = root + '/trn_val_split_json/train_split_0.json',
        pipeline = train_pipeline,
    ),
    val = dict(
        classes = classes,
        img_prefix = root,
        ann_file = root + '/trn_val_split_json/valid_split_0.json',
        pipeline = test_pipeline,
    ),
    test = dict(
        classes = classes,
        img_prefix = root,
        ann_file = root + 'test.json',
        pipeline = test_pipeline,
    ),
    samples_per_gpu=64,
    workers_per_gpu=4,
)

optimizer_config = dict(
    grad_clip = dict(
        _delete_=True,
        max_norm = 35,
        norm_type = 2
    )
)