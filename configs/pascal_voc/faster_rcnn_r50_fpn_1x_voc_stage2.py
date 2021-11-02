_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/vocstage2.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
data_root = '/home/qiucm/workspace/dataset/VOCdevkit/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
)
model = dict(
    roi_head=dict(
        type='MMStandardRoIHead',
        bbox_head=dict(
            type='MMShared2FCBBoxHead',
            num_classes=20,
            loss_mid_weight=0,
            loss_mem_cls_weight=0.2,
        ),
        memory_k=15827,
        top_k=32,
    ),
    train_cfg=dict(
        label_type2weight=[1,2,2],
        train_mod='ssod'
    ),
)
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[9])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 4 * 3 = 12
