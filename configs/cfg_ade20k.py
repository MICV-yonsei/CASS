_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_ade20k.txt',

    # CLIPb16 DINOb8
    global_semantics_weight = 0.3,
    mean_vector_weight = 0.05,
    h_threshold = 0.06

)

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
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
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
