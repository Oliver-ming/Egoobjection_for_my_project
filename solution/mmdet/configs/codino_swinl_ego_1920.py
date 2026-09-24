# High-resolution variant of codino_swinl_ego.py: short side 640..1088 (step 32), long side <= 1920,
# i.e. 1920x1080 frames are trained up to (and tested at) their native resolution.
_base_ = ['./codino_swinl_ego.py']

import copy

work_dir = '/home/zhangyuming/Egoobjection/solution/work_dirs/codino_swinl_ego_1920'

train_short_sides = list(range(640, 1089, 32))
train_max_side = 1920
test_scale = (1920, 1088)

_train_scales = [(s, train_max_side) for s in train_short_sides]
train_pipeline = copy.deepcopy(_base_.train_pipeline)
_choice = train_pipeline[3]
assert _choice['type'] == 'RandomChoice'
_choice['transforms'][0][0]['scales'] = _train_scales  # plain multi-scale branch
_choice['transforms'][1][2]['scales'] = _train_scales  # resize-crop-resize branch

test_pipeline = copy.deepcopy(_base_.test_pipeline)
test_pipeline[1]['scale'] = test_scale
test_pipeline_noann = [t for t in test_pipeline if t['type'] != 'LoadAnnotations']

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline_noann))
val_evaluator = dict(outfile_prefix=work_dir + '/preds/val')
test_evaluator = dict(outfile_prefix=work_dir + '/preds/test')

del copy, _train_scales, _choice
