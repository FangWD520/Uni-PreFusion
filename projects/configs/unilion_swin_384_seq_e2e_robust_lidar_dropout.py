_base_ = ['./unilion_swin_384_seq_e2e.py']

# LiDAR point dropout: randomly drop 50% points before depth generation.
robustness = dict(
    mode='lidar_dropout',
    cam_name='CAM_FRONT',
    dropout_ratio=0.5,
    trans_shift=[0.5, 0.0, 0.0],
    seed=0)
