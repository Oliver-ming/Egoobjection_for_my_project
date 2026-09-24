# Run test.py on the val set: prints COCO metrics AND writes <work_dir>/val.bbox.json (for WBF / analysis).
#   bash tools/dist_test.sh configs/dino_swinl_ego_val.py CKPT 4
# Other ann file / output: --cfg-options test_dataloader.dataset.ann_file=X test_evaluator.ann_file=X
#                                        test_evaluator.outfile_prefix=DIR/name
_base_ = './dino_swinl_ego.py'

test_dataloader = dict(dataset=dict(ann_file=_base_.val_ann, pipeline=_base_.val_pipeline))
test_evaluator = dict(ann_file=_base_.val_ann, format_only=False, classwise=True,
                      outfile_prefix=_base_.work_dir + '/val')
