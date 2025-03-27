_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_voc20.txt',

    # CLIPb16 DINOb8
    global_semantics_weight = 0.4,
    mean_vector_weight = 0.02,
    h_threshold = 0.10

)

# dataset settings
dataset_type = 'PascalVOC20Dataset'
data_root = ''

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
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))
