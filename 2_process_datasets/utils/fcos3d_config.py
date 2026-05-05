# FCOS3D Config for single image inference
# Based on: fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune

_base_ = [
    'mmdet3d::_base_/models/fcos3d.py',
    'mmdet3d::_base_/default_runtime.py'
]

# Model settings
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(
        num_classes=10,  # NuScenes classes
        bbox_code_size=9))

# Class names from NuScenes
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

metainfo = dict(classes=class_names)

# Test pipeline for single image
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(type='Pack3DDetInputs', keys=['img'])
]

# Dummy test_dataloader config (not actually used for single image inference)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Det3DDataset',
        data_root='.',
        ann_file='dummy.json',
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=metainfo))

test_evaluator = dict(type='KittiMetric', ann_file='dummy.json')
