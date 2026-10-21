_base_ = ['./unilion_swin_384_seq_perception.py']

# No pillar-to-voxel restoration: refined image features are used for LSS,
# while LiDAR sparse voxels keep their original features, i.e. \hat f_j = f_j.
model = dict(mm_frontend=dict(
    pillar_adapter='learned',
    restoration_strategy='none'))

load_from = 'ckpts/det+map_epoch_6.pth'

log_config = dict(hooks=[
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'),
    dict(
        type='WandbLoggerHook',
        init_kwargs=dict(
            project='UniLION-MambaFusion',
            name='ablation_restore_none_seq_perception',
            config=dict(
                config_file='projects/configs/ablation_restore_none_seq_perception.py',
                model='UniLION+MambaFusionMMBackbone',
                stage='seq_perception',
                restoration='none')))
])
