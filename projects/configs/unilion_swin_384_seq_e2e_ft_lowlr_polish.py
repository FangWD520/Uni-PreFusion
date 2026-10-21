_base_ = ['./unilion_swin_384_seq_e2e.py']

# Low-learning-rate polish config for the final multi-task stage.
# Goal: slightly improve det/map/planning from a strong 4-stage checkpoint
# without aggressively updating occ/motion.
# Usage:
#   tools/dist_train.sh projects/configs/unilion_swin_384_seq_e2e_ft_lowlr_polish.py 8 \
#       --cfg-options load_from=/path/to/your/best_stage4_checkpoint.pth

# Keep the model architecture unchanged.
# Adjust only the optimization schedule and task weights.
tasks = dict(
    det=True,
    occ=True,
    map=True,
    motion=True,
    planning=True,
)

optimizer = dict(
    type='AdamW',
    lr=1e-6,
    paramwise_cfg=dict(
        custom_keys={
            'mm_frontend.mf_img_backbone': dict(lr_mult=0.1),
            'mm_frontend.mf_img_neck.neck': dict(lr_mult=0.1),
            'det_head': dict(lr_mult=1.0),
            'map_head': dict(lr_mult=1.0),
            'occ_head': dict(lr_mult=0.1),
            'motion_head': dict(lr_mult=0.1),
            'planning_head': dict(lr_mult=1.0),
        }),
    weight_decay=0.05)

# Low-LR step decay polish schedule:
#   first half: 5e-6
#   second half: 1e-6
# gamma=0.2 makes 5e-6 -> 1e-6 at epoch total_epochs // 2.
lr_config = dict(_delete_=True, policy='step', step=[4], gamma=0.2)

total_epochs = 8
# Override the base IterBasedRunner with an epoch-based low-lr polish run.
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=total_epochs)

# Short fine-tuning stage.

checkpoint_config = dict(interval=1, max_keep_ckpts=4)

eval_mode = dict(
    with_det=tasks['det'],
    with_tracking=False,
    with_occ=tasks['occ'],
    with_map=tasks['map'],
    with_motion=tasks['motion'],
    with_planning=tasks['planning'],
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(interval=1, eval_mode=eval_mode)
# Full unfreezing increases activation/optimizer memory. Use batch size 1 per GPU
# to avoid OOM during polish fine-tuning.
data = dict(samples_per_gpu=1)

# Keep only a short polish run; the original fading schedule is not needed.
custom_hooks = []

# Balance the losses to favor det/map/planning while keeping occ/motion weakly
# regularized so they do not drift too much.
model = dict(
    train_cfg=dict(
        _delete_=True,
        dynamic=True,
        freeze=[],
        loss_det_weight=1.5,
        loss_occ_weight=0.1,
        loss_map_weight=0.5,
        loss_motion_weight=0.1,
        loss_planning_weight=1.0,
    )
)

# Load from your best stage-4 checkpoint, not resume.
load_from = None
resume_from = None
