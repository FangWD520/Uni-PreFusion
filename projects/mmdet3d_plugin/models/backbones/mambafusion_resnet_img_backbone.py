import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmdet.models.backbones import ResNet
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class MambaFusionResNetImageBackbone(nn.Module):
    """ResNet image pre-backbone for the MambaFusion-UniLION frontend.

    The original MambaFusion ``USE_PREBACKBONE=True`` path expects
    ``batch_dict['backbone_features']`` with shape [B*N, 128, H/8, W/8].
    This wrapper replaces the VMamba pre-backbone with ResNet while keeping
    the downstream MambaFusion mm_backbone and MF-Neck unchanged.
    """

    def __init__(self,
                 depth=50,
                 pretrained=None,
                 out_channels=128,
                 frozen_stages=0,
                 norm_eval=False,
                 with_cp=False,
                 style='pytorch',
                 **kwargs):
        super().__init__()
        self.pretrained = pretrained
        assert frozen_stages >= 0, 'Use frozen_stages>=0 because only two ResNet stages are constructed.'
        self.backbone = ResNet(
            depth=depth,
            num_stages=2,
            strides=(1, 2),
            dilations=(1, 1),
            out_indices=(1,),  # C3 / stride-8 feature, 512 channels for R50.
            stage_with_dcn=(False, False),
            frozen_stages=frozen_stages,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=norm_eval,
            with_cp=with_cp,
            style=style)
        stage_channels = 512 if depth >= 50 else 128
        self.proj = nn.Sequential(
            nn.Conv2d(stage_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def init_weights(self):
        if self.pretrained:
            load_checkpoint(self.backbone, self.pretrained, map_location='cpu', strict=False)
        else:
            self.backbone.init_weights()

    def train(self, mode=True):
        super().train(mode)
        # Keep mmdet ResNet's norm_eval behavior active after overriding train().
        if mode and self.backbone.norm_eval:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, img):
        """Forward raw multi-view images.

        Args:
            img: Tensor of shape [B, N, 3, H, W].

        Returns:
            Tensor of shape [B*N, out_channels, H/8, W/8].
        """
        b, n, c, h, w = img.shape
        x = img.reshape(b * n, c, h, w).float()
        feat = self.backbone(x)[0]
        return self.proj(feat)
