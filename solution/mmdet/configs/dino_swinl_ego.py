# DINO-5scale Swin-L (COCO 36e, 58.4 box AP) fine-tuned on the ego dataset (10 classes).
# Train (4 GPUs):  CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/dino_swinl_ego.py 4
# Train on train+val (final model): add
#   --cfg-options train_dataloader.dataset.ann_file=/home/zhangyuming/Egoobjection/solution/data/trainval.json
# Test set -> <work_dir>/test.bbox.json:  bash tools/dist_test.sh configs/dino_swinl_ego.py CKPT 4
# Val set  -> <work_dir>/val.bbox.json + metrics: use configs/dino_swinl_ego_val.py
_base_ = 'mmdet::dino/dino-5scale_swin-l_8xb2-36e_coco.py'

# `_base_/ego_dataset.py` cannot be a sibling base (both bases define `data_root`), so read it directly.
import runpy

_ego = runpy.run_path('{{ fileDirname }}/_base_/ego_dataset.py')
data_root = _ego['data_root']
metainfo = _ego['metainfo']
num_classes = _ego['num_classes']
train_ann = _ego['train_ann']
val_ann = _ego['val_ann']
test_ann = _ego['test_ann']
trainval_ann = _ego['trainval_ann']
img_prefix = _ego['img_prefix']
del _ego

work_dir = '/home/zhangyuming/Egoobjection/solution/work_dirs/dino_swinl_ego'
load_from = '/home/zhangyuming/Egoobjection/solution/weights/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'

batch_size = 2  # per GPU
max_epochs = 12
# COCO recipe: 1e-4 @ total batch 16. Linear rule -> 5e-5 @ 4 GPUs x 2. For other GPU counts pass
# --auto-scale-lr (rescales linearly from base_batch_size below).
lr = 5e-5

model = dict(
    # Swin ImageNet init is overwritten by load_from; None avoids an 800MB download in init_weights().
    backbone=dict(with_cp=True, init_cfg=None),
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(max_per_img=100))

# Standard DETR multi-scale aug (short side 480-800, long side <= 1333), same as the COCO recipe.
train_scales = [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333),
                (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)]
test_scale = (1333, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomChoiceResize', scales=train_scales, keep_ratio=True)],
            [
                dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)],
                     keep_ratio=True),
                dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600),
                     allow_negative_crop=True),
                dict(type='RandomChoiceResize', scales=train_scales, keep_ratio=True),
            ],
        ]),
    dict(type='PackDetInputs'),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=test_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=test_scale, keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=img_prefix,
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=img_prefix,
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=test_ann,
        data_prefix=img_prefix,
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(type='CocoMetric', ann_file=val_ann, metric='bbox', classwise=True,
                     format_only=False)
# Writes <outfile_prefix>.bbox.json (category_id 0..9, xywh in original image pixels).
test_evaluator = dict(type='CocoMetric', ann_file=test_ann, metric='bbox', format_only=True,
                      outfile_prefix=work_dir + '/test')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
auto_scale_lr = dict(enable=False, base_batch_size=8)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    # short warmup: the 10-class heads start from scratch
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[8, 11], gamma=0.1),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2,
                    save_best='coco/bbox_mAP', rule='greater'))
