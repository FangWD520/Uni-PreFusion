#!/usr/bin/env python3
import argparse
import os
from os import path as osp

import cv2
import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset
from tqdm import tqdm

from tools.visualization.cam_render import CamRender
from tools.visualization.bev_render import occ_color_map


def parse_args():
    parser = argparse.ArgumentParser('Camera-view qualitative visualization.')
    parser.add_argument('config')
    parser.add_argument('--baseline', default=None)
    parser.add_argument('--ours', required=True)
    parser.add_argument('--out-dir', default='vis/cam_fullstack')
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    parser.add_argument('--mode', choices=['det_plan', 'occ'], required=True)
    parser.add_argument('--resize-width', type=int, default=1200)
    parser.add_argument('--separate-cams', action='store_true',
                        help='Save each camera panel separately in addition to composed figures.')
    parser.add_argument('--no-grid', action='store_true',
                        help='Do not save composed camera grids; useful with --separate-cams.')
    parser.add_argument('--occ-source', choices=['ours', 'baseline', 'gt', 'all'], default='all',
                        help='Which occupancy result to project in --mode occ.')
    parser.add_argument('--occ-alpha', type=float, default=0.45,
                        help='Alpha for camera-view occupancy overlay.')
    parser.add_argument('--occ-stride', type=int, default=1,
                        help='Stride for non-free occupancy voxels before projection; increase if overlays are too dense.')
    parser.add_argument('--occ-max-points', type=int, default=120000,
                        help='Maximum non-free occupancy voxels projected per source; 0 disables sampling.')
    parser.add_argument('--occ-point-size', type=float, default=10.0,
                        help='Fallback projected voxel marker size in camera view.')
    parser.add_argument('--occ-render', choices=['box', 'point'], default='box',
                        help='Render camera-view occupancy as projected voxel boxes or point markers.')
    parser.add_argument('--occ-box-max-points', type=int, default=25000,
                        help='Maximum voxels rendered as projected boxes per source; above this, uniform sampling is applied.')
    parser.add_argument('--occ-blank-bg', action='store_true', default=True,
                        help='Render OCC projection on a blank white background instead of real camera images.')
    parser.add_argument('--occ-real-bg', dest='occ_blank_bg', action='store_false',
                        help='Use real camera images as background for OCC projection.')
    return parser.parse_args()


def get_plot_choices(mode):
    if mode == 'det_plan':
        return dict(draw_pred=True, det=True, occ=False, track=False, motion=False, map=False, planning=True)
    if mode == 'occ':
        return dict(draw_pred=True, det=False, occ=True, track=False, motion=False, map=False, planning=False)
    raise ValueError(mode)


def read_resize(path, width):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def add_title(img, title, height=44):
    canvas = np.ones((height + img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
    canvas[height:] = img
    cv2.putText(canvas, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def compose_rows(images, titles, out_path):
    titled = [add_title(img, t) for img, t in zip(images, titles)]
    max_w = max(im.shape[1] for im in titled)
    padded = []
    for im in titled:
        if im.shape[1] < max_w:
            pad = np.ones((im.shape[0], max_w - im.shape[1], 3), dtype=np.uint8) * 255
            im = np.hstack([im, pad])
        padded.append(im)
    grid = cv2.vconcat(padded)
    cv2.imwrite(out_path, grid)


def occ_to_points_colors(occ, stride=1, max_points=120000, zlim=(-1.0, 5.4)):
    occ = np.asarray(occ)
    free_id = 17
    mask = (occ != free_id) & (occ != 255)
    xs, ys, zs = np.where(mask)
    if stride > 1:
        xs, ys, zs = xs[::stride], ys[::stride], zs[::stride]
    labels = occ[xs, ys, zs].astype(np.int64)
    if max_points and xs.size > max_points:
        keep = np.linspace(0, xs.size - 1, max_points).astype(np.int64)
        xs, ys, zs, labels = xs[keep], ys[keep], zs[keep], labels[keep]
    h, w, d = occ.shape
    z_min, z_max = zlim
    pts = np.stack([
        (xs.astype(np.float32) + 0.5) / h * 80.0 - 40.0,
        (ys.astype(np.float32) + 0.5) / w * 80.0 - 40.0,
        z_min + (zs.astype(np.float32) + 0.5) / d * (z_max - z_min),
    ], axis=-1)
    colors = occ_color_map[np.clip(labels, 0, len(occ_color_map) - 1), :3] / 255.0
    return pts, colors


def occ_voxel_size_from_cfg(cfg, occ_shape):
    occ_head = cfg.model.get('occ_head', {}) if hasattr(cfg, 'model') else {}
    output_scope = occ_head.get('bev_occ_config', {}).get('output_scope', None)
    h, w, d = occ_shape
    if output_scope is not None:
        dx = float(output_scope[0][2])
        dy = float(output_scope[1][2])
    else:
        dx, dy = 80.0 / h, 80.0 / w
    dz = 6.4 / d
    return dx, dy, dz


def voxel_corners(points, voxel_size):
    dx, dy, dz = voxel_size
    offsets = np.asarray([
        [-dx / 2, -dy / 2, -dz / 2], [dx / 2, -dy / 2, -dz / 2],
        [dx / 2, dy / 2, -dz / 2], [-dx / 2, dy / 2, -dz / 2],
        [-dx / 2, -dy / 2, dz / 2], [dx / 2, -dy / 2, dz / 2],
        [dx / 2, dy / 2, dz / 2], [-dx / 2, dy / 2, dz / 2],
    ], dtype=np.float32)
    return points[:, None, :] + offsets[None, :, :]


def load_gt_occ(data):
    return np.load(data['ann_info']['occ_path'])['semantics']


def render_occ_overlay(data, occ, out_path, cfg, title='OCC', alpha=0.45, stride=1, max_points=120000,
                       point_size=10.0, render='box', box_max_points=25000, blank_bg=True):
    from tools.visualization.cam_render import CAM_NAMES_NUSC, CAM_NAMES_NUSC_converter
    from nuscenes.utils.geometry_utils import view_points
    from matplotlib.collections import PolyCollection
    import matplotlib.pyplot as plt
    pts, colors = occ_to_points_colors(occ, stride=stride, max_points=max_points)
    if pts.shape[0] == 0:
        return False
    voxel_size = occ_voxel_size_from_cfg(cfg, np.asarray(occ).shape)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for j, cam in enumerate(CAM_NAMES_NUSC):
        idx = CAM_NAMES_NUSC_converter.index(cam)
        raw_img = cv2.imread(data['img_filename'][idx])
        img_h, img_w = raw_img.shape[:2]
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255 if blank_bg else cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        ax = axes[j // 3, j % 3]
        ax.imshow(img)
        ax.axis('off')
        cam_intrinsic = np.asarray(data['cams'][cam]['cam_intrinsic'])
        extrinsic = np.asarray(data['cams'][cam]['lidar2cam'])
        cam_pts = pts @ extrinsic[:3, :3] + extrinsic[3, :3]
        valid = cam_pts[:, 2] > 0.1
        if valid.any():
            proj_colors = colors[valid]
            depth = cam_pts[valid, 2]
            pts_valid = pts[valid]
            if render == 'box':
                if box_max_points and pts_valid.shape[0] > box_max_points:
                    keep = np.linspace(0, pts_valid.shape[0] - 1, box_max_points).astype(np.int64)
                    pts_valid, proj_colors, depth = pts_valid[keep], proj_colors[keep], depth[keep]
                corners = voxel_corners(pts_valid, voxel_size).reshape(-1, 3)
                cam_corners = corners @ extrinsic[:3, :3] + extrinsic[3, :3]
                positive = cam_corners[:, 2] > 0.1
                proj = np.zeros((cam_corners.shape[0], 2), dtype=np.float32)
                proj[positive] = view_points(cam_corners[positive].T, cam_intrinsic, normalize=True)[:2].T
                proj = proj.reshape(-1, 8, 2)
                positive = positive.reshape(-1, 8)
                order = np.argsort(depth)[::-1]
                polys, face_colors = [], []
                for k in order:
                    if not positive[k].all():
                        continue
                    poly = proj[k]
                    x0, y0 = poly[:, 0].min(), poly[:, 1].min()
                    x1, y1 = poly[:, 0].max(), poly[:, 1].max()
                    if x1 < 0 or x0 >= img.shape[1] or y1 < 0 or y0 >= img.shape[0]:
                        continue
                    polys.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                    face_colors.append(proj_colors[k])
                if polys:
                    pc = PolyCollection(polys, facecolors=face_colors, edgecolors='none', alpha=alpha)
                    ax.add_collection(pc)
            else:
                proj = view_points(cam_pts[valid].T, cam_intrinsic, normalize=True)[:2].T
                m = ((proj[:, 0] >= 0) & (proj[:, 0] < img.shape[1]) &
                     (proj[:, 1] >= 0) & (proj[:, 1] < img.shape[0]))
                proj, depth, proj_colors = proj[m], depth[m], proj_colors[m]
                if proj.shape[0] > 0:
                    order = np.argsort(depth)[::-1]
                    ax.scatter(proj[order, 0], proj[order, 1], s=point_size,
                               c=proj_colors[order], alpha=alpha, marker='s', linewidths=0)
        ax.set_title(cam, fontsize=9)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    ours = mmcv.load(args.ours)
    baseline = mmcv.load(args.baseline) if args.baseline else None
    if args.mode == 'det_plan':
        save_grid = not args.no_grid
        ours_render = CamRender(get_plot_choices('det_plan'), osp.join(args.out_dir, 'ours'),
                                save_separate_cams=args.separate_cams, save_grid=save_grid)
        base_render = CamRender(get_plot_choices('det_plan'), osp.join(args.out_dir, 'baseline'),
                                save_separate_cams=args.separate_cams, save_grid=save_grid) if baseline is not None else None
    for idx in tqdm(args.indices):
        data = dataset.get_data_info(idx)
        token = data.get('token', str(idx))
        if args.mode == 'det_plan':
            imgs = []
            titles = []
            if base_render is not None:
                base_path = base_render.render_pred_gt_overlay(
                    data, baseline[idx], idx, name='Baseline', pred_color=np.asarray([0.12, 0.35, 1.0]))
                imgs.append(read_resize(base_path, args.resize_width))
                titles.append('Baseline vs GT: Detection + Planning')
            ours_path = ours_render.render_pred_gt_overlay(
                data, ours[idx], idx, name='Ours', pred_color=np.asarray([0.95, 0.10, 0.10]))
            imgs.append(read_resize(ours_path, args.resize_width))
            titles.append('Ours vs GT: Detection + Planning')
            if not args.no_grid:
                out_path = osp.join(args.out_dir, f'{idx:04d}_{token}_cam_det_plan.jpg')
                compose_rows(imgs, titles, out_path)
        else:
            sources = []
            if args.occ_source in ['baseline', 'all'] and baseline is not None:
                sources.append(('baseline', np.asarray(baseline[idx]['pts_occ']['occ'])))
            if args.occ_source in ['ours', 'all']:
                sources.append(('ours', np.asarray(ours[idx]['pts_occ']['occ'])))
            if args.occ_source in ['gt', 'all']:
                sources.append(('gt', load_gt_occ(data)))
            imgs, titles = [], []
            for name, occ in sources:
                tmp_path = osp.join(args.out_dir, f'{idx:04d}_{token}_cam_occ_{name}.jpg')
                render_occ_overlay(data, occ, tmp_path, cfg, title=f'{name} semantic occupancy',
                                   alpha=args.occ_alpha, stride=args.occ_stride,
                                   max_points=args.occ_max_points, point_size=args.occ_point_size,
                                   render=args.occ_render, box_max_points=args.occ_box_max_points,
                                   blank_bg=args.occ_blank_bg)
                imgs.append(read_resize(tmp_path, args.resize_width))
                titles.append(f'{name.upper()} camera-view occupancy')
            out_path = osp.join(args.out_dir, f'{idx:04d}_{token}_cam_occ.jpg')
            compose_rows(imgs, titles, out_path)
            print('saved', out_path)


if __name__ == '__main__':
    main()
