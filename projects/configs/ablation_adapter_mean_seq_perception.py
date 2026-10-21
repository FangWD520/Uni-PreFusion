_base_ = ['./unilion_swin_384_seq_perception.py']

model = dict(mm_frontend=dict(pillar_adapter='mean'))
load_from = 'ckpts/unilion_384_seq_perception_from_det_map_init.pth'
