# Co-DINO (5-scale, Swin-L) fine-tuned from the Objects365->COCO checkpoint on the ego dataset.
#
# Train (4 GPUs x 1 img):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash solution/mmdet/tools/dist_train.sh \
#       solution/mmdet/configs/codino_swinl_ego.py 4
# Train on train+val: use codino_swinl_ego_trainval.py, or add
#   --cfg-options train_dataloader.dataset.ann_file=/home/zhangyuming/Egoobjection/solution/data/trainval.json
# Val / test inference -> <prefix>.bbox.json: solution/mmdet/tools/dist_infer.sh (see README_codino.md).
_base_ = ['../projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py']

import copy

from mmengine.config import Config

_ego = Config.fromfile('{{ fileDirname }}/_base_/ego_dataset.py')

# ----------------------------------------------------------------------------------------------
# Main knobs
# ----------------------------------------------------------------------------------------------
work_dir = '/home/zhangyuming/Egoobjection/solution/work_dirs/codino_swinl_ego'
load_from = '/home/zhangyuming/Egoobjection/solution/weights/' \
            'co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
pretrained = None  # backbone ImageNet weights are not needed: everything comes from `load_from`

train_ann_file = _ego.train_ann  # _ego.trainval_ann for the final model (see *_trainval.py)

# Resolution: DETR-style multi-scale, short side 480..800 (step 32), long side <= 1333.
# 1920x1080 frames are trained at 853x480 .. 1333x750 and tested at 1333x750.
train_short_sides = list(range(480, 801, 32))
train_max_side = 1333
test_scale = (1333, 800)

# Schedule (total batch = num_gpus x 1).  Original: AdamW lr 1e-4 @ bs16, backbone x0.1.
# bs4 -> 5e-5 (square-root scaling, the usual rule for Adam-type optimizers; linear scaling
# would give 2.5e-5 which under-trains the re-initialised class layers in 12 epochs).
base_lr = 5e-5
max_epochs = 12
lr_decay_epochs = [10]
warmup_iters = 500
val_interval = 1

num_classes = _ego.num_classes
metainfo = _ego.metainfo
data_root = _ego.data_root
dataset_type = 'CocoDataset'
backend_args = None

# ----------------------------------------------------------------------------------------------
# Model: every classifying head -> 10 classes
# ----------------------------------------------------------------------------------------------
_roi_head = copy.deepcopy(_base_.model.roi_head)
_roi_head[0]['bbox_head']['num_classes'] = num_classes
_bbox_head = copy.deepcopy(_base_.model.bbox_head)
_bbox_head[0]['num_classes'] = num_classes
_test_cfg = copy.deepcopy(_base_.model.test_cfg)
_test_cfg[0]['max_per_img'] = 100  # query head (the one used at inference); COCO maxDets=100
_train_cfg = copy.deepcopy(_base_.model.train_cfg)
# AMP: with >= split_thr (default 10000) boxes mmcv's batched_nms takes a per-level loop that
# index-puts fp32 scores into an fp16 tensor and crashes. The aux RPN keeps <= 6 x 4000 boxes;
# the single-call (coordinate-offset) path is numerically identical and dtype-safe.
_train_cfg[1]['rpn_proposal']['nms']['split_thr'] = 100000

model = dict(
    backbone=dict(init_cfg=None),  # with_cp=True already set in the base config
    query_head=dict(num_classes=num_classes),  # also sizes the DN label embedding
    roi_head=_roi_head,
    bbox_head=_bbox_head,
    train_cfg=_train_cfg,
    test_cfg=_test_cfg)

# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
_train_scales = [(s, train_max_side) for s in train_short_sides]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomChoiceResize', scales=_train_scales, keep_ratio=True)],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(type='RandomChoiceResize', scales=_train_scales, keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=test_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline_noann = [t for t in test_pipeline if t['type'] != 'LoadAnnotations']

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=_ego.img_prefix,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=_ego.val_ann,
        data_prefix=_ego.img_prefix,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=_ego.test_ann,
        data_prefix=_ego.img_prefix,
        test_mode=True,
        pipeline=test_pipeline_noann,
        backend_args=backend_args))

# Validation also dumps <work_dir>/preds/val.bbox.json (overwritten at every evaluation).
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=_ego.val_ann,
    metric='bbox',
    classwise=True,
    outfile_prefix=work_dir + '/preds/val',
    backend_args=backend_args)
# Test set has no GT: only write <work_dir>/preds/test.bbox.json.
test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=_ego.test_ann,
    metric='bbox',
    classwise=True,  # used when tools/dist_infer.sh evaluates a json that has GT
    format_only=True,
    outfile_prefix=work_dir + '/preds/test',
    backend_args=backend_args)

# ----------------------------------------------------------------------------------------------
# Optimisation / schedule / runtime
# ----------------------------------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=lr_decay_epochs,
        gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,  # last epoch only (+ the best one below, kept separately)
        save_best='coco/bbox_mAP',
        rule='greater'))

# `--auto-scale-lr` would rescale `base_lr` linearly w.r.t. this batch size.
auto_scale_lr = dict(enable=False, base_batch_size=4)

del Config, copy, _ego, _roi_head, _bbox_head, _test_cfg, _train_cfg, _train_scales
