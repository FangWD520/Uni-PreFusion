_base_ = ['./unilion_swin_384_seq_e2e.py']

# Missing front-view camera trio.
robustness = dict(
    mode='missing_front_camera',
    cam_names=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
    dropout_ratio=0.5,
    trans_shift=[0.5, 0.0, 0.0],
    yaw_deg=0.0,
    seed=0)
