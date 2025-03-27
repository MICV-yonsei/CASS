_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_coco_object.txt',
    prob_thd=0.1,

    # CLIPb16 DINOb8
    global_semantics_weight = 0.25,
    mean_vector_weight = 0.01,
    h_threshold = 0.10

)

# dataset settings
dataset_type = 'COCOObjectDataset'
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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))
