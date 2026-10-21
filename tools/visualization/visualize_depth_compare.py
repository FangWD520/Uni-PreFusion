import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from projects.mmdet3d_plugin.datasets.pipelines.loading import PrepareImageInputs

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import projects.mmdet3d_plugin  # noqa: F401
import projects.mmdet3d_plugin.datasets  # noqa: F401
import projects.mmdet3d_plugin.models  # noqa: F401

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def unwrap(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        return unwrap(x[0]) if len(x) > 0 else x
    if hasattr(x, 'data'):
        return unwrap(x.data)
    return x


def denorm_rgb(img):
    img = img.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return np.clip(img[..., ::-1], 0, 255).astype(np.uint8)


def depth_to_map(depth_prob, depth_cfg, out_hw):
    dmin, dmax, step = depth_cfg
    bins = torch.arange(dmin, dmax, step, device=depth_prob.device)[:depth_prob.shape[0]]
    depth = (depth_prob * bins.view(-1, 1, 1)).sum(0)
    depth = F.interpolate(depth[None, None], size=out_hw, mode='bilinear', align_corners=False)[0, 0]
    return depth.cpu().numpy()


def load_model(cfg, ckpt):
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt, map_location='cpu')
    model.eval()
    if torch.cuda.is_available():
        model = MMDataParallel(model.cuda(), device_ids=[0])
    return model


def capture_front_depth(model, batch):
    store = {}
    core = model.module if hasattr(model, 'module') else model

    def hook(_, __, out):
        store['depth'] = out[3].detach().cpu()

    handle = core.img_map2bev.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(return_loss=False, rescale=True, **batch)
    handle.remove()

    img = unwrap(batch['img'])
    if img.dim() == 5:
        rgb = denorm_rgb(img[0, 0])
    elif img.dim() == 4:
        rgb = denorm_rgb(img[0])
    else:
        rgb = denorm_rgb(img)

    depth = store['depth']
    if depth.dim() == 5:
        depth = depth[0, 0]
    elif depth.dim() == 4:
        depth = depth[0]
    if depth.dim() != 3:
        raise RuntimeError(f'Unexpected depth shape after squeeze: {tuple(depth.shape)}')
    gt = unwrap(batch['gt_depth'])
    if gt.dim() == 4:
        gt = gt[0, 0]
    elif gt.dim() == 3:
        gt = gt[0]
    gt = gt.float().cpu().numpy()
    return rgb, gt, depth


def save_png(arr, out_path, cmap=None, vmin=None, vmax=None, dpi=600):
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if cmap is not None:
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(color='white')
    else:
        cm = None
    ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def valid_range(*arrays, default=(1, 60)):
    vals = []
    for arr in arrays:
        m = np.isfinite(arr) & (arr > 0)
        if m.any():
            vals.append(arr[m])
    if not vals:
        return default
    vals = np.concatenate(vals)
    return max(default[0], float(np.percentile(vals, 2))), min(default[1], float(np.percentile(vals, 98)))


def save_depth_overlay(rgb, gt, out_path, vmin, vmax, dpi=600):
    h, w = gt.shape
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rgb)
    ys, xs = np.where(gt > 0)
    if len(xs) > 0:
        ax.scatter(xs, ys, c=gt[ys, xs], s=0.8, cmap='turbo', vmin=vmin, vmax=vmax, linewidths=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_sample_pngs(out_dir, idx, token, rgb, gt, pred):
    prefix = Path(out_dir) / f'{idx:06d}_{token}'
    mask = gt > 0
    gt_vis = np.where(mask, gt, np.nan)
    err = np.full_like(pred, np.nan, dtype=np.float32)
    err[mask] = np.abs(pred[mask] - gt[mask])
    pred_vmin, pred_vmax = valid_range(pred, default=(1, 60))
    sparse_vmin, sparse_vmax = valid_range(gt_vis, np.where(mask, pred, np.nan))
    save_png(rgb, str(prefix) + '_rgb.png')
    save_depth_overlay(rgb, gt, str(prefix) + '_gt_overlay.png', sparse_vmin, sparse_vmax)
    save_png(gt_vis, str(prefix) + '_gt_depth_auto.png', cmap='turbo', vmin=sparse_vmin, vmax=sparse_vmax)
    save_png(pred, str(prefix) + '_pred_depth_dense.png', cmap='turbo', vmin=pred_vmin, vmax=pred_vmax)
    save_png(np.where(mask, pred, np.nan), str(prefix) + '_pred_depth_on_gt.png', cmap='turbo', vmin=sparse_vmin, vmax=sparse_vmax)
    save_png(err, str(prefix) + '_abs_error_auto.png', cmap='inferno', vmin=0, vmax=np.nanpercentile(err, 95) if np.isfinite(err).any() else 20)


def valid_mae(pred, gt):
    mask = gt > 0
    return float(np.abs(pred[mask] - gt[mask]).mean()) if mask.any() else float('nan')


def render_compare(base_npz, ours_npz, out_path):
    base = np.load(base_npz)
    ours = np.load(ours_npz)
    rgb, gt = base['rgb'], base['gt']
    bpred, opred = base['pred'], ours['pred']
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    panels = [
        (rgb, None, 'RGB'),
        (gt, 'magma', 'GT depth'),
        (bpred, 'magma', f'Baseline MAE={valid_mae(bpred, gt):.2f}'),
        (opred, 'magma', f'Ours MAE={valid_mae(opred, gt):.2f}'),
    ]
    for ax, (arr, cmap, title) in zip(axs, panels):
        ax.imshow(arr, cmap=cmap, vmin=1 if cmap else None, vmax=60 if cmap else None)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config for dump mode')
    parser.add_argument('--checkpoint', help='checkpoint for dump mode')
    parser.add_argument('--indices', type=int, nargs='*', default=[0, 25, 50])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--compare', nargs=2, metavar=('BASE_DIR', 'OURS_DIR'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.compare:
        base_dir, ours_dir = map(Path, args.compare)
        for base_npz in sorted(base_dir.glob('*.npz')):
            ours_npz = ours_dir / base_npz.name
            if ours_npz.exists():
                render_compare(base_npz, ours_npz, Path(args.out_dir) / base_npz.name.replace('.npz', '.png'))
        return

    cfg = Config.fromfile(args.config)
    cfg.data.val.work_dir = cfg.get('work_dir', args.out_dir)
    dataset = build_dataset(cfg.data.val)
    model = load_model(cfg, args.checkpoint)
    raw_dataset = dataset
    data_info_dataset = build_dataset(cfg.data.test)
    if hasattr(dataset, 'dataset'):
        raw_dataset = dataset.dataset
    image_pipeline = PrepareImageInputs(data_config=cfg.data_config, is_train=False)
    for idx in args.indices:
        if idx < 0 or idx >= len(raw_dataset):
            print(f'skip invalid index {idx}, dataset length={len(raw_dataset)}')
            continue
        sample = data_info_dataset.get_data_info(idx)
        sample['test_mode'] = True
        sample = image_pipeline(sample)
        batch = collate([sample], samples_per_gpu=1)
        if torch.cuda.is_available():
            batch = scatter(batch, [0])[0]
        rgb, gt, depth_prob = capture_front_depth(model, batch)
        pred = depth_to_map(depth_prob, cfg.grid_config['depth'], gt.shape)
        metas = unwrap(batch['img_metas'])
        if isinstance(metas, dict):
            token = metas['token']
        else:
            token = metas[0]['token']
        save_sample_pngs(args.out_dir, idx, token, rgb, gt, pred)
        print('saved pngs', idx, token)


if __name__ == '__main__':
    main()
