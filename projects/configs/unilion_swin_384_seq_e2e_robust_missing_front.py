_base_ = ['./unilion_swin_384_seq_e2e.py']

# Missing front camera: zero CAM_FRONT image and projected depth map.
robustness = dict(
    mode='missing_front_camera',
    cam_name='CAM_FRONT',
    dropout_ratio=0.5,
    trans_shift=[0.5, 0.0, 0.0],
    seed=0)
