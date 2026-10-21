_base_ = ['./unilion_swin_384_seq_e2e.py']

model = dict(mm_frontend=dict(pillar_adapter='learned'))
load_from = 'ckpts/resnet_2.pth'
