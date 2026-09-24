# Final-model variant: identical to codino_swinl_ego.py but trained on train+val
# (solution/data/trainval.json). Val metrics are then optimistic (val is in training) and only a
# sanity check, so no "best" checkpoint is kept: use the last epoch.
# For the high-resolution variant, change `_base_` to './codino_swinl_ego_1920.py'.
_base_ = ['./codino_swinl_ego.py']

from mmengine.config import Config

_ego = Config.fromfile('{{ fileDirname }}/_base_/ego_dataset.py')

work_dir = '/home/zhangyuming/Egoobjection/solution/work_dirs/codino_swinl_ego_trainval'
train_ann_file = _ego.trainval_ann

train_dataloader = dict(dataset=dict(ann_file=train_ann_file))
val_evaluator = dict(outfile_prefix=work_dir + '/preds/val')
test_evaluator = dict(outfile_prefix=work_dir + '/preds/test')

train_cfg = dict(val_interval=4)
default_hooks = dict(checkpoint=dict(save_best=None, max_keep_ckpts=1))

del Config, _ego
