# dataset settings

_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/default_runtime.py'
]
dataset_type = 'GHDDataset'
data_root = '/mnt/d/Dataset/ghd'
tr_img_prefix = 'train'
classes = None
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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
model=dict(
    roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=14, sample_num=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        pipeline=train_pipeline,
        data_root = data_root,
        img_prefix = tr_img_prefix,
        ann_file='fold0_train.pkl',
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        pipeline=test_pipeline,
        ann_file='fold0_val.pkl',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        pipeline=test_pipeline,
        ann_file='fold0_val.pkl',
        ))