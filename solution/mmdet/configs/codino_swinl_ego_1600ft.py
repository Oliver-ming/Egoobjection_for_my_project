# Short high-resolution fine-tune from the 1333 epoch-12 train-only checkpoint.
# Medium objects are the weak spot (APm 0.72 vs APl 0.90); train closer to native 1920x1080.
_base_ = ['./codino_swinl_ego.py']

import copy

work_dir = '/home/zhangyuming/Egoobjection/solution/work_dirs/codino_swinl_ego_1600ft'
load_from = '/home/zhangyuming/Egoobjection/solution/work_dirs/codino_swinl_ego/epoch_12.pth'

train_short_sides = list(range(800, 1201, 32))
train_max_side = 1600
test_scale = (1600, 960)

_train_scales = [(s, train_max_side) for s in train_short_sides]
train_pipeline = copy.deepcopy(_base_.train_pipeline)
_choice = train_pipeline[3]
assert _choice['type'] == 'RandomChoice'
_choice['transforms'][0][0]['scales'] = _train_scales
_choice['transforms'][1][2]['scales'] = _train_scales

test_pipeline = copy.deepcopy(_base_.test_pipeline)
test_pipeline[1]['scale'] = test_scale
test_pipeline_noann = [t for t in test_pipeline if t['type'] != 'LoadAnnotations']

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline_noann))
val_evaluator = dict(outfile_prefix=work_dir + '/preds/val')
test_evaluator = dict(outfile_prefix=work_dir + '/preds/test')

max_epochs = 6
base_lr = 1e-5  # weights are already converged at 5e-5; this stage only adapts to the larger scale
optim_wrapper = dict(optimizer=dict(lr=base_lr))
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[4], gamma=0.1),
]
auto_scale_lr = dict(enable=False, base_batch_size=8)

del copy, _train_scales, _choice
