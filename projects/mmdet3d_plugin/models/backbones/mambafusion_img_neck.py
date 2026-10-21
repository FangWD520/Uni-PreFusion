import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from mmdet.models.builder import BACKBONES

from pcdet.models.backbones_image.img_neck.generalized_lss import MY_FPN


@BACKBONES.register_module()
class MambaFusionImageNeck(nn.Module):
    """MambaFusion image neck wrapper for UniLION.

    It applies the MambaFusion FPN to multi-scale mm_backbone image features,
    concatenates all image_fpn outputs along channel dimension, and optionally
    projects them to the channel count expected by UniLION LSS.
    """

    def __init__(self,
                 in_channels=(128, 128),
                 fpn_out_channels=256,
                 out_channels=256,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 use_bias=True,
                 align_corners=True):
        super().__init__()
        cfg = EasyDict(
            IN_CHANNELS=list(in_channels),
            OUT_CHANNELS=fpn_out_channels,
            NUM_OUTS=num_outs,
            START_LEVEL=start_level,
            END_LEVEL=end_level,
            USE_BIAS=use_bias,
            ALIGN_CORNERS=align_corners)
        self.neck = MY_FPN(cfg)
        used_levels = len(in_channels) - 1 if end_level == -1 else end_level - start_level - 1
        used_levels = max(1, used_levels)
        merged_channels = fpn_out_channels * used_levels
        self.out_channels = out_channels
        self.proj = nn.Identity() if merged_channels == out_channels else nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, image_features):
        if len(image_features) == 1:
            return image_features[0]
        target_size = image_features[0].shape[2:]
        outs = list(self.neck(image_features))
        outs = [feat if feat.shape[2:] == target_size else F.interpolate(
            feat, size=target_size, mode='bilinear', align_corners=False) for feat in outs]
        merged = torch.cat(outs, dim=1)
        return self.proj(merged)
