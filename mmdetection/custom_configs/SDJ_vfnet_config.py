# model settings
model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100))

load_from = "https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth"

#################
#### dataset ####
#################

dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ['General trash','Paper','Paper pack','Metal','Glass','Plastic',
'Styrofoam','Plastic bag','Battery','Clothing']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


albu_train_transforms = [
    dict(type='HorizontalFlip',p=0.3),
    dict(
        type="OneOf",
        transforms=[
            dict(type="HueSaturationValue", hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
            dict(type="RandomGamma"),
         #   dict(type="CLAHE"),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="MotionBlur"),
            dict(type="GaussNoise"),
        ],
        p=0.4,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RandomToneCurve", scale=1.5),
            dict(type='RandomBrightnessContrast',brightness_limit=[-0.2, 0.2],contrast_limit=[-0.2, 0.2]),
        ],
        p=0.4,
    ),
    dict(type='RGBShift', p = 0.2)
]


train_pipeline = [
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='MixUp',img_scale=(1024, 1024),ratio_range=(0.8, 1.2),pad_val=114.0, p=0.2),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
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
    samples_per_gpu=8, # batch size
    workers_per_gpu=2, # data loader
    train=dict(
        type='MultiImageMixDataset',# type=dataset_type,
        dataset =dict(
        type = dataset_type,
        ann_file=data_root + "trn_val_split_json/train_split_0.json",#'trn_val_split_json/train_split_0.json',
        img_prefix=data_root,
        pipeline= [
                dict(type='LoadImageFromFile', to_float32=False),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
        classes = classes,
        ),
    pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "trn_val_split_json/valid_split_0.json",#'trn_val_split_json/valid_split_0.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline))
    
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50',classwise = True)

###############
## scheduler ##
###############

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[16, 22])

lr_config = dict(
    policy='CosineAnnealing', by_epoch=False, warmup='linear',
    warmup_iters=500, warmup_ratio=0.001, min_lr=1e-07)
runner = dict(type='EpochBasedRunner', max_epochs=24)

#
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project="dj_project",
                name=f'mask-rcnn-x101-32x4d_fpn', 
                entity="cval"
                ))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]

work_dir = './work_dirs/rm_outlier'
seed = 42
fp16 = dict(loss_scale=512.)