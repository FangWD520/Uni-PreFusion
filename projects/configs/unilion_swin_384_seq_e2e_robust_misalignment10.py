_base_ = ['./unilion_swin_384_seq_e2e.py']

robustness = dict(
    mode='camera_lidar_misalignment',
    cam_name='CAM_FRONT',
    dropout_ratio=0.5,
    trans_shift=[1.0, 0.0, 0.0],
    yaw_deg=10.0,
    seed=0)
