#!/usr/bin/env python3
"""Extract one-batch Uni-DenseFusion intermediate feature heatmaps.

Example:
python tools/visualization/extract_fusion_heatmaps.py \
  projects/configs/unilion_swin_384_seq_e2e.py ckpts/unilion_swin_384_seq_e2e.pth \
  --out-dir vis/fusion_heatmaps --sample-index 0 --cam-id 0
"""
import argparse
import importlib
import os
from pathlib import Path
from types import MethodType
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch_scatter
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.models import build_detector

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--out-dir', default='vis/fusion_heatmaps')
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--cam-id', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def import_plugins(cfg, config_path):
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if getattr(cfg, 'plugin', False):
        if hasattr(cfg, 'plugin_dir'):
            module_dir = os.path.dirname(cfg.plugin_dir).split('/')
        else:
            module_dir = os.path.dirname(config_path).split('/')
        module_path = module_dir[0]
        for m in module_dir[1:]:
            module_path += '.' + m
        importlib.import_module(module_path)


def build_one_batch(cfg, sample_index):
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = 1
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1,
                              workers_per_gpu=0, dist=False, shuffle=False)
    for i, data in enumerate(loader):
        if i == sample_index:
            return data
    raise IndexError(f'sample_index {sample_index} out of range')


def tensor_cpu(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu()
    if isinstance(x, (list, tuple)):
        return [tensor_cpu(v) for v in x]
    return x


def normalize(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return a


def feature_heatmap(feat, cam_id=0):
    """Convert [B,N,C,H,W], [B*N,C,H,W], or [B,C,H,W] to a 2D heatmap."""
    if feat is None:
        return None
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    feat = tensor_cpu(feat)
    if feat.ndim == 5:
        x = feat[0, min(cam_id, feat.shape[1] - 1)]
    elif feat.ndim == 4:
        x = feat[0]
    else:
        return None
    return normalize(x.abs().mean(0).numpy())


def sparse_bev_raw(features, coords, grid_hw=None, mode='norm'):
    features, coords = tensor_cpu(features), tensor_cpu(coords).long()
    if features is None or coords is None or features.numel() == 0:
        return None
    mask = coords[:, 0] == 0
    features, coords = features[mask], coords[mask]
    if coords.numel() == 0:
        return None
    y, x = coords[:, 2], coords[:, 3]
    h = int(y.max().item() + 1) if grid_hw is None else grid_hw[0]
    w = int(x.max().item() + 1) if grid_hw is None else grid_hw[1]
    score = features.norm(dim=1) if mode == 'norm' else features.abs().mean(dim=1)
    linear = y.clamp(0, h - 1) * w + x.clamp(0, w - 1)
    bev = torch_scatter.scatter_max(score, linear, dim=0, dim_size=h * w)[0]
    return np.rot90(bev.reshape(h, w).numpy(), k=-1)


def sparse_bev_heatmap(features, coords, grid_hw=None):
    bev = sparse_bev_raw(features, coords, grid_hw)
    return None if bev is None else normalize(bev)


def difference_heatmap(after, before):
    if after is None or before is None:
        return None
    h, w = min(after.shape[0], before.shape[0]), min(after.shape[1], before.shape[1])
    diff = after[:h, :w] - before[:h, :w]
    scale = np.percentile(np.abs(diff), 99)
    return np.clip(diff / max(scale, 1e-6), -1, 1)


def save_img(path, arr, cmap='magma'):
    if arr is None:
        return
    plt.figure(figsize=(4, 4), dpi=300)
    plt.imshow(arr, cmap=cmap)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_rgb_reference(path, data, cam_id=0):
    img_dc = data.get('img', None)
    if img_dc is None:
        return
    img = img_dc.data[0] if hasattr(img_dc, 'data') else img_dc
    if isinstance(img, (list, tuple)):
        img = img[0]
    if not torch.is_tensor(img):
        return
    img = img.detach().cpu().float()
    if img.ndim == 5:
        im = img[0, min(cam_id, img.shape[1] - 1)]
    elif img.ndim == 4:
        im = img[min(cam_id, img.shape[0] - 1)]
    else:
        return
    im = im.permute(1, 2, 0).numpy()
    im = normalize(im)
    plt.figure(figsize=(5, 3), dpi=300)
    plt.imshow(im)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_rgb_array(data, cam_id=0):
    img_dc = data.get('img', None)
    if img_dc is None:
        return None
    img = img_dc.data[0] if hasattr(img_dc, 'data') else img_dc
    if isinstance(img, (list, tuple)):
        img = img[0]
    if not torch.is_tensor(img):
        return None
    img = img.detach().cpu().float()
    if img.ndim == 5:
        im = img[0, min(cam_id, img.shape[1] - 1)]
    elif img.ndim == 4:
        im = img[min(cam_id, img.shape[0] - 1)]
    else:
        return None
    im = im.permute(1, 2, 0).numpy()
    return normalize(im)


def save_feature_overlay(path, rgb, heat, alpha=0.45):
    if rgb is None or heat is None:
        return
    plt.figure(figsize=(5, 3), dpi=300)
    plt.imshow(rgb)
    plt.imshow(heat, cmap='jet', alpha=alpha, extent=[0, rgb.shape[1], rgb.shape[0], 0])
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def voxel_world_coords(coords, voxel_size=(0.3, 0.3, 0.25), pc_range=(-54.0, -54.0, -5.0)):
    coords = tensor_cpu(coords).long()
    if coords is None or coords.numel() == 0:
        return None
    mask = coords[:, 0] == 0
    coords = coords[mask]
    if coords.numel() == 0:
        return None
    z, y, x = coords[:, 1].float(), coords[:, 2].float(), coords[:, 3].float()
    wx = x * voxel_size[0] + pc_range[0] + voxel_size[0] * 0.5
    wy = y * voxel_size[1] + pc_range[1] + voxel_size[1] * 0.5
    wz = z * voxel_size[2] + pc_range[2] + voxel_size[2] * 0.5
    return torch.stack([wx, wy, wz], dim=1)


def save_voxel_3d(path, coords, scores=None, title=None, max_points=6000, cmap='inferno'):
    xyz = voxel_world_coords(coords)
    if xyz is None:
        return
    xyz = xyz.numpy()
    if scores is None:
        values = xyz[:, 2]
    else:
        values = tensor_cpu(scores)
        values = values[coords[:, 0].long() == 0].numpy()
    if len(xyz) > max_points:
        if scores is None:
            idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        else:
            idx = np.argsort(values)[-max_points:]
        xyz, values = xyz[idx], values[idx]
    values = normalize(values)
    fig = plt.figure(figsize=(5.0, 4.0), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=values, s=2.0, cmap=cmap, alpha=0.85, linewidths=0)
    ax.view_init(elev=28, azim=-55)
    ax.set_xlim(-54, 54); ax.set_ylim(-54, 54); ax.set_zlim(-5, 3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 0.22))
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_alpha(0.0)
    if title:
        ax.set_title(title, fontsize=9, pad=0)
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close()



def install_mm_frontend_capture(model, cache):
    mmf = getattr(model, 'mm_frontend', None)
    if mmf is None:
        raise RuntimeError('model.mm_frontend is None; no pre-lifting fusion module found')
    old_forward = mmf.forward

    def wrapped_forward(self, *args, **kwargs):
        cache['pts_voxel_before'] = tensor_cpu(kwargs.get('pts_voxel_features'))
        cache['pts_coords_before'] = tensor_cpu(kwargs.get('pts_voxel_coords'))
        if kwargs.get('feature_maps') is not None:
            cache['img_feat_before'] = tensor_cpu(kwargs.get('feature_maps'))
        out = old_forward(*args, **kwargs)
        cache['img_feat_after'] = tensor_cpu(out.get('feature_maps'))
        cache['pts_voxel_after'] = tensor_cpu(out.get('pts_voxel_features'))
        cache['pts_coords_after'] = tensor_cpu(out.get('pts_voxel_coords'))
        for name in ['vis_cache', '_vis_cache', 'fusion_vis_cache']:
            if hasattr(self, name):
                cache[name] = getattr(self, name)
        return out


    if getattr(mmf, 'mf_img_backbone', None) is not None:
        def img_backbone_hook(module, inputs, output):
            cache['img_feat_before'] = tensor_cpu(output)
        mmf.mf_img_backbone.register_forward_hook(img_backbone_hook)

    mmf.forward = MethodType(wrapped_forward, mmf)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    data = build_one_batch(cfg, args.sample_index)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device).eval()

    cache = {}
    install_mm_frontend_capture(model, cache)
    model = MMDataParallel(model, device_ids=[int(args.device.split(':')[-1])])
    with torch.no_grad():
        model(return_loss=False, rescale=True, **data)

    torch.save(cache, out_dir / 'fusion_feature_cache.pt')
    save_rgb_reference(out_dir / 'rgb_reference.png', data, args.cam_id)
    save_img(out_dir / 'image_feature_after.png', feature_heatmap(cache.get('img_feat_after'), args.cam_id), 'jet')
    save_img(out_dir / 'image_feature_before.png', feature_heatmap(cache.get('img_feat_before'), args.cam_id), 'jet')
    rgb = get_rgb_array(data, args.cam_id)
    cam_after = feature_heatmap(cache.get('img_feat_after'), args.cam_id)
    cam_before = feature_heatmap(cache.get('img_feat_before'), args.cam_id)
    save_feature_overlay(out_dir / 'cam_feature_after_fusion_overlay.png', rgb, cam_after)
    save_feature_overlay(out_dir / 'cam_feature_before_fusion_overlay.png', rgb, cam_before)


    bev_before_raw = sparse_bev_raw(cache.get('pts_voxel_before'), cache.get('pts_coords_before'))
    bev_after_raw = sparse_bev_raw(cache.get('pts_voxel_after'), cache.get('pts_coords_after'))
    save_img(out_dir / 'lidar_bev_before.png', None if bev_before_raw is None else normalize(bev_before_raw), 'magma')
    save_img(out_dir / 'lidar_bev_after.png', None if bev_after_raw is None else normalize(bev_after_raw), 'magma')
    save_img(out_dir / 'lidar_bev_delta_signed.png', difference_heatmap(bev_after_raw, bev_before_raw), 'coolwarm')
    save_img(out_dir / 'lidar_bev_delta_abs.png', None if difference_heatmap(bev_after_raw, bev_before_raw) is None else np.abs(difference_heatmap(bev_after_raw, bev_before_raw)), 'inferno')

    save_voxel_3d(out_dir / 'voxel_before_3d.png', cache.get('pts_coords_before'),
                  scores=None, title=None, max_points=8000, cmap='viridis')
    if cache.get('pts_voxel_before') is not None and cache.get('pts_voxel_after') is not None:
        b, a = cache['pts_voxel_before'], cache['pts_voxel_after']
        cb, ca = cache.get('pts_coords_before'), cache.get('pts_coords_after')
        n = min(b.shape[0], a.shape[0], cb.shape[0], ca.shape[0])
        # Coordinates are expected to be preserved by residual broadcast. Use feature delta as voxel color.
        delta_score = (a[:n] - b[:n]).norm(dim=1)
        save_voxel_3d(out_dir / 'voxel_update_3d.png', cb[:n], scores=delta_score,
                      title=None, max_points=5000, cmap='inferno')


    if cache.get('pts_voxel_before') is not None and cache.get('pts_voxel_after') is not None:
        b, a = cache['pts_voxel_before'], cache['pts_voxel_after']
        n = min(b.shape[0], a.shape[0])
        rel = (a[:n] - b[:n]).norm(dim=1).mean() / (b[:n].norm(dim=1).mean() + 1e-6)
        print(f'Mean relative LiDAR feature update: {float(rel):.4f}')
    print(f'Saved feature cache and heatmaps to: {out_dir}')


if __name__ == '__main__':
    main()
