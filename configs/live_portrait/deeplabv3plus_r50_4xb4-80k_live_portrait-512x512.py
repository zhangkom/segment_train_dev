crop_size = (512, 512)
data_root = 'data/live_portrait'

metainfo = dict(
    classes=('background', 'person'),
    palette=[[0, 0, 0], [0, 255, 0]],
)

dataset_type = 'BaseSegDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='masks/train'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
        reduce_zero_label=False,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='masks/val'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        reduce_zero_label=False,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024_20200605_161823-7ba4f549.pth'
resume = False

tta_model = dict(type='SegTTAModel')

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
    ),
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
        ],
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1000, end=80000, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

randomness = dict(seed=3407)

launcher = 'none'
work_dir = './work_dirs/live_portrait_deeplabv3plus'

