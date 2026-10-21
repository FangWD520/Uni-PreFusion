import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from projects.mmdet3d_plugin.models.backbones.mambafusion_stage12_frontend import (  # noqa: E402
    MambaFusionStage12FrontEnd,
)


def main():
    torch.manual_seed(0)

    model = MambaFusionStage12FrontEnd(
        img_in_channels=256,
        pts_in_channels=128,
        hidden_channels=128,
        img_pool_stride=4,
    )
    model.eval()

    b, n, c, h, w = 2, 6, 256, 32, 88
    nv = 2000
    feature_maps = [torch.randn(b, n, c, h, w)]
    pts_voxel_features = torch.randn(nv, 128)
    pts_voxel_coords = torch.zeros(nv, 4, dtype=torch.long)
    pts_voxel_coords[: nv // 2, 0] = 0
    pts_voxel_coords[nv // 2 :, 0] = 1
    pts_voxel_coords[:, 1] = torch.randint(0, 4, (nv,))
    pts_voxel_coords[:, 2] = torch.randint(0, 360, (nv,))
    pts_voxel_coords[:, 3] = torch.randint(0, 360, (nv,))

    with torch.no_grad():
        outputs = model(
            feature_maps=feature_maps,
            pts_voxel_features=pts_voxel_features,
            pts_voxel_coords=pts_voxel_coords,
        )

    fused_feat = outputs['feature_maps'][0]
    fused_pts = outputs['pts_voxel_features']
    fused_coords = outputs['pts_voxel_coords']

    print('input image feat:', tuple(feature_maps[0].shape))
    print('output image feat:', tuple(fused_feat.shape))
    print('input lidar voxel feat:', tuple(pts_voxel_features.shape))
    print('output lidar voxel feat:', tuple(fused_pts.shape))
    print('output lidar voxel coords:', tuple(fused_coords.shape))

    assert fused_feat.shape == feature_maps[0].shape
    assert fused_pts.shape == pts_voxel_features.shape
    assert fused_coords.shape == pts_voxel_coords.shape
    print('Smoke test passed.')


if __name__ == '__main__':
    main()

