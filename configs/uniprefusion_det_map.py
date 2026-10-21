"""Uni-PreFusion detection and map training stage."""
_base_ = ['../projects/configs/unilion_swin_384_det_map.py']
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
work_dir = 'outputs/Uni-PreFusion/det_map'
load_from = None
resume_from = None
