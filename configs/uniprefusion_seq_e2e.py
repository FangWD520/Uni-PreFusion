"""Uni-PreFusion final five-task configuration; preserve checkpoint keys."""
_base_ = ['../projects/configs/unilion_swin_384_seq_e2e.py']

# Exact checkpoint anchor tensors, exported as constructor resources.
model = dict(
    motion_head=dict(motion_anchor='assets/kmeans_motion_6.npy'),
    planning_head=dict(planning_anchor='assets/kmeans_planning_6.npy'))
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
work_dir = 'outputs/Uni-PreFusion/seq_e2e'
load_from = None
resume_from = None
