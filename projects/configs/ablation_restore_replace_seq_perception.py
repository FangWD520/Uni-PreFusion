_base_ = ['./unilion_swin_384_seq_perception.py']

# Direct broadcast / replacement: every voxel in a pillar receives the same
# refined pillar token, i.e. \hat f_j = \hat f_{p(j)}.
model = dict(mm_frontend=dict(
    pillar_adapter='learned',
    restoration_strategy='replace'))

load_from = 'ckpts/det+map_epoch_6.pth'

log_config = dict(hooks=[
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'),
    dict(
        type='WandbLoggerHook',
        init_kwargs=dict(
            project='UniLION-MambaFusion',
            name='ablation_restore_replace_seq_perception',
            config=dict(
                config_file='projects/configs/ablation_restore_replace_seq_perception.py',
                model='UniLION+MambaFusionMMBackbone',
                stage='seq_perception',
                restoration='replace')))
])
