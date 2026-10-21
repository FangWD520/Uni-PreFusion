_base_ = ['./unilion_swin_384_seq_e2e.py']

# Image-backbone ablation: replace VMamba pre-backbone with ResNet50 while
# keeping the same MambaFusion mm_backbone path and MF-Neck.
model = dict(
    mm_frontend=dict(
        mf_img_backbone=dict(
            _delete_=True,
            type='MambaFusionResNetImageBackbone',
            depth=50,
            pretrained='ckpts/resnet50-19c8e357.pth',
            out_channels=128,
            frozen_stages=0,
            norm_eval=False,
            with_cp=False,
            style='pytorch'),
        mf_img_neck=dict(
            type='MambaFusionImageNeck',
            in_channels=(128, 128),
            fpn_out_channels=256,
            out_channels=256,
            num_outs=4,
            start_level=0,
            end_level=-1,
            use_bias=True,
            align_corners=True)))
