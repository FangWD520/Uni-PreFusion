_base_ = ['./unilion_swin_384_det_map.py']

# Low-learning-rate fine-tuning config for det+map stage.
# Usage:
#   tools/dist_train.sh projects/configs/unilion_swin_384_det_map_ft_lowlr_2ep.py 8 \
#       --cfg-options load_from=/path/to/your/epoch_6.pth
#
# This config uses load_from, not resume_from. It loads only model weights from
# the 6-epoch checkpoint and starts a fresh 2-epoch low-lr fine-tuning run.

optimizer = dict(
    type='AdamW',
    lr=1e-6,
    paramwise_cfg=dict(
        custom_keys={
            'mm_frontend.mf_img_backbone': dict(lr_mult=0.1),
            'mm_frontend.mf_img_neck.neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.05)

# Keep the learning rate fixed at the low fine-tuning value.
# _delete_ avoids inheriting cyclic-scheduler-only keys from the base config.
lr_config = dict(_delete_=True, policy='fixed')

# Run 2 extra epochs on top of the loaded 6-epoch checkpoint.
total_epochs = 6
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

# Disable the original late-stage Fading hook; it is designed for the 12-epoch
# schedule and is not needed for a 2-epoch low-lr fine-tune.
custom_hooks = []

# Evaluate every fine-tuning epoch.
eval_mode = dict(
    with_det=True,
    with_tracking=False,
    with_occ=False,
    with_map=True,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(interval=2, eval_mode=eval_mode)

# Set this via --cfg-options load_from=/path/to/epoch_6.pth, or edit here.
load_from = None
resume_from = None
