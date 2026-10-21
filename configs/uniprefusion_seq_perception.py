"""Uni-PreFusion temporal perception: detection, map and occupancy."""
_base_ = ['../projects/configs/unilion_swin_384_seq_perception.py']
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
work_dir = 'outputs/Uni-PreFusion/seq_perception'
load_from = None
resume_from = None
