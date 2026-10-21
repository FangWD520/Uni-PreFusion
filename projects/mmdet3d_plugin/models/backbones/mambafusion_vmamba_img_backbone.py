import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mmdet.models.builder import BACKBONES

from pcdet.models.vmamba.vmamba import Backbone_VSSM


@BACKBONES.register_module()
class MambaFusionVMambaImageBackbone(nn.Module):
    """Standalone VMamba image backbone extracted from MambaFusion mm_backbone.

    The module mirrors the VMamba branch used by the original MambaFusion
    mm_backbone and outputs a single feature map consumed through
    ``batch_dict['backbone_features']`` when mm_backbone is set to
    ``USE_PREBACKBONE=True``.
    """

    def __init__(self,
                 image_size=(256, 704),
                 out_channels=128,
                 depths=(2, 2),
                 out_indices=(1,),
                 dims=128,
                 ssm_d_state=1,
                 drop_path_rate=0.2,
                 with_cp=True):
        super().__init__()
        self.image_size = list(image_size)
        self.out_channels = out_channels
        self.with_cp = with_cp
        self.backbone_vssm = Backbone_VSSM(
            norm_layer='ln2d',
            patch_size=4,
            in_chans=3,
            depths=tuple(depths),
            out_indices=list(out_indices),
            dims=dims,
            ssm_d_state=ssm_d_state,
            ssm_conv_bias=False,
            forward_type='v05_noz',
            drop_path_rate=drop_path_rate,
            downsample_version='v3',
            patchembed_version='v2')
        self.down_block = nn.Conv2d(dims * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        """Forward raw multi-view images.

        Args:
            img: Tensor of shape [B, N, 3, H, W].

        Returns:
            Tensor of shape [B*N, C, H/8, W/8].
        """
        b, n, c, h, w = img.shape
        x = img.reshape(b * n, c, h, w).float()
        if self.training and self.with_cp:
            feats = checkpoint(self.backbone_vssm, x, use_reentrant=False)
        else:
            feats = self.backbone_vssm(x)
        feat = feats[0]
        feat = self.down_block(feat)
        return feat
