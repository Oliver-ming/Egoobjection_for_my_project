# Dataset constants shared by all MMDetection configs of this solution.
# Import in a config with:  _base_ = ['./_base_/ego_dataset.py', ...]  and reference via {{_base_.xxx}}
# or simply copy the values; pipelines are model-specific and defined in each config.
data_root = '/home/zhangyuming/Egoobjection/resource/data/'
solution_data_root = '/home/zhangyuming/Egoobjection/solution/data/'

# Names must match `categories[].name` in the json files exactly (ids 0..9 in this order).
metainfo = dict(
    classes=('pillow', 'box', 'book', 'bottle', 'chair', 'mug', 'door', 'shelf', 'plate', 'sofa'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)],
)
num_classes = 10

train_ann = data_root + 'train.json'
val_ann = data_root + 'val.json'
test_ann = data_root + 'test.json'
trainval_ann = solution_data_root + 'trainval.json'  # produced by solution/tools/prepare_data.py
img_prefix = dict(img=data_root + 'images/')
