"""Uni-PreFusion detection training stage."""
_base_ = ['../projects/configs/unilion_swin_384_det.py']
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
work_dir = 'outputs/Uni-PreFusion/det'
load_from = None
resume_from = None
