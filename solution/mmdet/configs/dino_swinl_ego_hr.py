# Higher-resolution variant: short side 576-1024, long side <= 2048 (1920x1080 -> 1820x1024 at test).
_base_ = './dino_swinl_ego.py'

work_dir = '/home/zhangyuming/Egoobjection/solution/work_dirs/dino_swinl_ego_hr'

train_scales = [(576, 2048), (608, 2048), (640, 2048), (672, 2048), (704, 2048), (736, 2048),
                (768, 2048), (800, 2048), (832, 2048), (864, 2048), (896, 2048), (928, 2048),
                (960, 2048), (992, 2048), (1024, 2048)]
test_scale = (2048, 1024)
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_evaluator = dict(outfile_prefix=work_dir + '/test')
