import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from projects.mmdet3d_plugin.models.backbones.mambafusion_mm_backbone_frontend import (  # noqa: E402
    MambaFusionMMBackboneFrontEnd,
)


def _state_dict(path):
    ckpt = torch.load(path, map_location='cpu')
    return ckpt.get('model_state', ckpt.get('state_dict', ckpt.get('model', ckpt)))


def main():
    ckpt_path = os.path.join(ROOT, 'ckpts/checkpoint_epoch_10.pth')
    state = _state_dict(ckpt_path)
    expected = [k for k in state if k.startswith('mm_backbone.')]

    model = MambaFusionMMBackboneFrontEnd(mf_checkpoint=ckpt_path)
    if hasattr(model.mm_backbone, 'load_mambafusion_state'):
        loaded, missing, mismatch = model.mm_backbone.load_mambafusion_state(state)
    else:
        mapped = {k[len('mm_backbone.'):]: v for k, v in state.items() if k.startswith('mm_backbone.')}
        ret = model.mm_backbone.load_state_dict(mapped, strict=False)
        loaded = [k for k in mapped if k not in ret.unexpected_keys]
        missing = ret.missing_keys
        mismatch = []
    print('expected mm_backbone tensors:', len(expected))
    print('loaded mm_backbone tensors:', len(loaded))
    print('missing:', len(missing))
    print('mismatch:', len(mismatch))
    assert len(loaded) == len(expected), (len(loaded), len(expected), mismatch[:5])
    assert not missing
    assert not mismatch

    b, n, h, w = 1, 6, 256, 704
    nv = 512
    img = torch.randn(b, n, 3, h, w)
    pts_voxel_features = torch.randn(nv, 128)
    pts_voxel_coords = torch.zeros(nv, 4, dtype=torch.long)
    pts_voxel_coords[:, 1] = torch.randint(0, 4, (nv,))
    pts_voxel_coords[:, 2] = torch.randint(0, 360, (nv,))
    pts_voxel_coords[:, 3] = torch.randint(0, 360, (nv,))

    model.eval()
    with torch.no_grad():
        out = model(
            img=img,
            pts_voxel_features=pts_voxel_features,
            pts_voxel_coords=pts_voxel_coords,
        )
    print('raw img:', tuple(img.shape))
    print('fused image feat:', tuple(out['feature_maps'][0].shape))
    print('fused lidar feat:', tuple(out['pts_voxel_features'].shape))
    assert out['feature_maps'][0].shape[:3] == (b, n, 256)
    assert out['pts_voxel_features'].shape == pts_voxel_features.shape
    print('MambaFusion mm_backbone load/forward smoke test passed.')


if __name__ == '__main__':
    main()
