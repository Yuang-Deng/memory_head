_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/vocstage1.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
data_root = '/data/dya/dataset/VOCdevkit/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
)
model = dict(
    type='EMAFasterRCNN',
    rpn_head=dict(
        anchor_generator=dict(
            scales=[8,16,32]
        )
    ),
    roi_head=dict(
        type='MMStandardRoIHead',
        contrastive_lambda2=0.2,
        contrastive_lambda1=0.2,
        unlabel_contrastive_lambda2=0.2,
        ori_pos_k=1,
        warm_epoch=0,
        memory_k=65536,
        pos_k=1,
        T1=0.2,
        T2=0.2,
        ema=0.99,
        pseudo_gen_hook=False,
        ctr_dim=128,
        bbox_head=dict(
            type='MMShared2FCBBoxHead',
            num_classes=20,
            loss_mid_weight=0,
        )
    ),
    train_cfg=dict(
        label_type2weight=[1,2,2],
        ema=0.999,
        rcnn=dict(
            sampler=dict(
                num=256,
            )
        ),
        ctr_rpn_assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.7,
            match_low_quality=True,
            ignore_iof_thr=-1),
    ),
)
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='MEMEMAHook')]
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[9])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 4 * 3 = 12
