_base_ = ['./unilion_swin_384_seq_e2e.py']

# E2E adapter ablation: learned sparse-to-pillar pooling without height embedding.
# Load from the corresponding learned_no_height seq_perception checkpoint, e.g.:
# --cfg-options load_from=outputs/<learned_no_height_seq_perception_run>/latest.pth
model = dict(mm_frontend=dict(
    pillar_adapter='learned_no_height',
    restoration_strategy='residual'))

load_from = None

log_config = dict(hooks=[
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'),
    dict(
        type='WandbLoggerHook',
        init_kwargs=dict(
            project='UniLION-MambaFusion',
            name='ablation_adapter_learned_no_height_seq_e2e',
            config=dict(
                config_file='projects/configs/ablation_adapter_learned_no_height_seq_e2e.py',
                model='UniLION+MambaFusionMMBackbone',
                stage='seq_e2e',
                adapter='learned_no_height_pillar_pool')))
])
