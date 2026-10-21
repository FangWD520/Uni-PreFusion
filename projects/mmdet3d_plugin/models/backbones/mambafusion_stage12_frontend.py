import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from projects.mmdet3d_plugin.models.utils.mamba_compat import ExternalMambaBlock
from mmdet.models.builder import BACKBONES


class ResidualMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_ratio=2):
        super().__init__()
        hidden = max(out_channels, int(out_channels * hidden_ratio))
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, x):
        return self.net(x)


@BACKBONES.register_module()
class MambaFusionStage12FrontEnd(nn.Module):
    """Lightweight MambaFusion-style stage1/stage2 front-end for UniLION.

    It keeps UniLION's data flow intact while injecting two rounds of residual
    camera-lidar fusion before LSS and Lion3DBackbone.
    """

    def __init__(self,
                 img_in_channels=256,
                 pts_in_channels=128,
                 hidden_channels=128,
                 img_pool_stride=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 drop_path=0.0):
        super().__init__()
        self.img_pool_stride = img_pool_stride
        self.img_to_hidden = nn.Linear(img_in_channels, hidden_channels)
        self.hidden_to_img = nn.Linear(hidden_channels, img_in_channels)

        self.stage1_img_norm = nn.LayerNorm(hidden_channels)
        self.stage2_img_norm = nn.LayerNorm(hidden_channels)
        self.stage1_img_block = ExternalMambaBlock(
            d_model=hidden_channels, d_state=d_state, d_conv=d_conv,
            expand=expand, drop_path=drop_path, layer_id=0, n_layer=2,
            with_cp=False)
        self.stage2_img_block = ExternalMambaBlock(
            d_model=hidden_channels, d_state=d_state, d_conv=d_conv,
            expand=expand, drop_path=drop_path, layer_id=1, n_layer=2,
            with_cp=False)

        self.stage1_img_cross = nn.Sequential(nn.LayerNorm(pts_in_channels), nn.Linear(pts_in_channels, hidden_channels))
        self.stage2_img_cross = nn.Sequential(nn.LayerNorm(pts_in_channels), nn.Linear(pts_in_channels, hidden_channels))
        self.stage1_pts_cross = ResidualMLP(pts_in_channels + hidden_channels, pts_in_channels)
        self.stage2_pts_cross = ResidualMLP(pts_in_channels + hidden_channels, pts_in_channels)

    def _pool_image_tokens(self, feat):
        b, n, c, h, w = feat.shape
        flat = feat.reshape(b * n, c, h, w)
        pooled = F.avg_pool2d(flat, self.img_pool_stride, self.img_pool_stride)
        _, _, hp, wp = pooled.shape
        tokens = pooled.view(b, n, c, hp, wp).permute(0, 1, 3, 4, 2).reshape(b, n * hp * wp, c)
        return tokens, (hp, wp)

    def _recover_delta(self, tokens, b, n, c, h, w, hp, wp):
        delta = self.hidden_to_img(tokens)
        delta = delta.view(b, n, hp, wp, c).permute(0, 1, 4, 2, 3).reshape(b * n, c, hp, wp)
        delta = F.interpolate(delta, size=(h, w), mode='bilinear', align_corners=False)
        return delta.view(b, n, c, h, w)

    def _pts_batch_summary(self, pts_voxel_features, batch_ids, batch_size):
        if pts_voxel_features.numel() == 0:
            return pts_voxel_features.new_zeros(batch_size, pts_voxel_features.shape[-1])
        return torch_scatter.scatter_mean(pts_voxel_features, batch_ids, dim=0, dim_size=batch_size)

    def _fuse_stage(self, img_tokens, pts_voxel_features, batch_ids, batch_size,
                    img_norm, img_block, img_cross, pts_cross):
        img_summary = img_tokens.mean(dim=1)
        pts_summary = self._pts_batch_summary(pts_voxel_features, batch_ids, batch_size)

        img_tokens = img_tokens + img_cross(pts_summary).unsqueeze(1)
        img_tokens = img_tokens + img_block(img_norm(img_tokens))

        refreshed_img_summary = img_tokens.mean(dim=1)
        pts_cond = refreshed_img_summary[batch_ids]
        pts_update = pts_cross(torch.cat([pts_voxel_features, pts_cond], dim=-1))
        pts_voxel_features = pts_voxel_features + pts_update
        return img_tokens, pts_voxel_features

    def forward(self,
                feature_maps,
                pts_voxel_features,
                pts_voxel_coords,
                img_inputs=None,
                img_metas=None):
        if not feature_maps or pts_voxel_features is None or pts_voxel_coords is None:
            return dict(
                feature_maps=feature_maps,
                pts_voxel_features=pts_voxel_features,
                pts_voxel_coords=pts_voxel_coords,
            )

        primary_feat = feature_maps[0]
        b, n, c, h, w = primary_feat.shape
        batch_ids = pts_voxel_coords[:, 0].long()

        img_tokens, (hp, wp) = self._pool_image_tokens(primary_feat)
        img_tokens = self.img_to_hidden(img_tokens)

        img_tokens, pts_voxel_features = self._fuse_stage(
            img_tokens, pts_voxel_features, batch_ids, b,
            self.stage1_img_norm, self.stage1_img_block,
            self.stage1_img_cross, self.stage1_pts_cross)
        img_tokens, pts_voxel_features = self._fuse_stage(
            img_tokens, pts_voxel_features, batch_ids, b,
            self.stage2_img_norm, self.stage2_img_block,
            self.stage2_img_cross, self.stage2_pts_cross)

        fused_feature_maps = list(feature_maps)
        fused_feature_maps[0] = primary_feat + self._recover_delta(img_tokens, b, n, c, h, w, hp, wp)
        return dict(
            feature_maps=fused_feature_maps,
            pts_voxel_features=pts_voxel_features,
            pts_voxel_coords=pts_voxel_coords,
        )

