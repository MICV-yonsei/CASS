_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_coco_stuff.txt',

    # CLIPb16 DINOb8
    global_semantics_weight = 0.2,
    mean_vector_weight = 0.02,
    h_threshold = 0.02

)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = '/nfs/home/chanyoung/kochanha/freeda/src/data/coco_stuff164k/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))
