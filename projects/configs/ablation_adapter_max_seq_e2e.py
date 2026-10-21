_base_ = ['./unilion_swin_384_seq_e2e.py']

# E2E adapter ablation: max sparse-to-pillar pooling.
# Load from the corresponding max-adapter seq_perception checkpoint, e.g.:
# --cfg-options load_from=outputs/<max_seq_perception_run>/latest.pth
model = dict(mm_frontend=dict(
    pillar_adapter='max',
    restoration_strategy='residual'))

load_from = None

log_config = dict(hooks=[
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'),
    dict(
        type='WandbLoggerHook',
        init_kwargs=dict(
            project='UniLION-MambaFusion',
            name='ablation_adapter_max_seq_e2e',
            config=dict(
                config_file='projects/configs/ablation_adapter_max_seq_e2e.py',
                model='UniLION+MambaFusionMMBackbone',
                stage='seq_e2e',
                adapter='max_pillar_pool')))
])
