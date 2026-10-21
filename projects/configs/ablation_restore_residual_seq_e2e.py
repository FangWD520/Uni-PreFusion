_base_ = ['./unilion_swin_384_seq_e2e.py']

# Ours: residual pillar-to-voxel broadcast for the frozen perception frontend.
model = dict(mm_frontend=dict(
    pillar_adapter='learned',
    restoration_strategy='residual'))

# Set this to the corresponding seq-perception checkpoint after training.
load_from = None

log_config = dict(hooks=[
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'),
    dict(
        type='WandbLoggerHook',
        init_kwargs=dict(
            project='UniLION-MambaFusion',
            name='ablation_restore_residual_seq_e2e',
            config=dict(
                config_file='projects/configs/ablation_restore_residual_seq_e2e.py',
                model='UniLION+MambaFusionMMBackbone',
                stage='seq_e2e',
                restoration='residual')))
])
