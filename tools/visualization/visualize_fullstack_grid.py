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

from tools.visualization.bev_render import BEVRender


ROW_CHOICES = {
    'Detection + Motion + Planning': dict(draw_pred=True, det=True, occ=False, track=False, motion=True, map=False, planning=True),
    'Map': dict(draw_pred=True, det=False, occ=False, track=False, motion=False, map=True, planning=False),
    'Occupancy': dict(draw_pred=True, det=False, occ=True, track=False, motion=False, map=False, planning=False),
}
COL_TITLES = ['Baseline', 'Ours', 'GT / Reference']


def parse_args():
    parser = argparse.ArgumentParser('Compose 3x3 full-stack qualitative figures.')
    parser.add_argument('config')
    parser.add_argument('--baseline', required=True, help='Baseline results.pkl')
    parser.add_argument('--ours', required=True, help='Ours results.pkl')
    parser.add_argument('--out-dir', default='vis/fullstack_grid')
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    parser.add_argument('--resize-width', type=int, default=640)
    parser.add_argument('--overlay-det-motion-planning', action='store_true',
                        help='Add a single overlay row comparing Baseline/Ours/GT with different colors.')
    parser.add_argument('--pair-gt-overlay', action='store_true',
                        help='Add two panels: Baseline vs GT and Ours vs GT.')
    parser.add_argument('--overlay-only', action='store_true',
                        help='Only render the selected overlay comparison row.')
    parser.add_argument('--motion-horizon', type=int, default=6,
                        help='Number of agent future steps to draw. UniLION predicts 12 steps; use 6 for a compact, planning-matched horizon.')
    parser.add_argument('--planning-horizon', type=int, default=6,
                        help='Number of ego future steps to draw.')
    parser.add_argument('--occ-view', choices=['bev', 'oblique', 'open3d'], default='bev',
                        help='Render occupancy as BEV, Matplotlib oblique scatter, or STCOCC-style Open3D voxels.')
    parser.add_argument('--occ-elev', type=float, default=62.0,
                        help='Elevation angle for 3D occupancy views.')
    parser.add_argument('--occ-azim', type=float, default=-90.0,
                        help='Azimuth angle for 3D occupancy views.')
    parser.add_argument('--occ-marker-size', type=float, default=2.0,
                        help='Marker size for oblique occupancy voxels.')
    parser.add_argument('--occ-alpha', type=float, default=0.9,
                        help='Marker alpha for oblique occupancy voxels.')
    parser.add_argument('--occ-max-points', type=int, default=60000,
                        help='Maximum non-free voxels to draw per occupancy panel; set 0 to disable sampling.')
    parser.add_argument('--occ-open3d-view-json', default=None,
                        help='Optional Open3D pinhole camera parameter json, e.g. STCOCC view.json.')
    parser.add_argument('--no-occ-open3d-car', action='store_true',
                        help='Do not draw the black ego car box in --occ-view open3d.')
    parser.add_argument('--occ-open3d-wait-time', type=float, default=0.1,
                        help='Seconds to update the offscreen Open3D renderer before screenshot.')
    parser.add_argument('--occ-focus-range', type=float, default=42.0,
                        help='Half range shown by fallback 3D OCC view; smaller values zoom in around ego.')
    parser.add_argument('--export-occ', action='store_true',
                        help='Export GT/baseline/ours occupancy arrays for external Open3D/STCOCC viewing.')
    parser.add_argument('--export-occ-ply', action='store_true',
                        help='Also export colored point-cloud PLY files for direct Open3D inspection.')
    return parser.parse_args()


def add_title(img, title, height=42):
    canvas = np.ones((height + img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
    canvas[height:] = img
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def read_and_resize(path, width):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def pad_to_same_height(images):
    max_h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] == max_h:
            padded.append(im)
            continue
        pad = np.ones((max_h - im.shape[0], im.shape[1], 3), dtype=np.uint8) * 255
        padded.append(np.vstack([im, pad]))
    return padded


def compose_grid(row_imgs, out_path):
    rows = []
    for row_name, imgs in row_imgs:
        titled = [add_title(img, title) for img, title in zip(imgs, COL_TITLES)]
        titled = pad_to_same_height(titled)
        row = cv2.hconcat(titled)
        row = add_title(row, row_name, height=46)
        rows.append(row)
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.ones((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8) * 255
            r = np.hstack([r, pad])
        padded_rows.append(r)
    grid = cv2.vconcat(padded_rows)
    cv2.imwrite(out_path, grid)


def render_row(row_name, choices, data, base_res, ours_res, idx, out_dir, width, motion_horizon,
               planning_horizon, occ_kwargs=None):
    row_dir = osp.join(out_dir, row_name.replace(' ', '_').replace('+', 'plus'))
    occ_kwargs = occ_kwargs or {}
    base_renderer = BEVRender(choices, osp.join(row_dir, 'baseline'),
                              motion_horizon=motion_horizon, planning_horizon=planning_horizon,
                              **occ_kwargs)
    ours_renderer = BEVRender(choices, osp.join(row_dir, 'ours'),
                              motion_horizon=motion_horizon, planning_horizon=planning_horizon,
                              **occ_kwargs)
    base_gt, base_pred = base_renderer.render(data, base_res, idx)
    _, ours_pred = ours_renderer.render(data, ours_res, idx)
    imgs = [
        read_and_resize(base_pred, width),
        read_and_resize(ours_pred, width),
        read_and_resize(base_gt, width),
    ]
    return imgs


def render_overlay_row(data, base_res, ours_res, idx, out_dir, width, motion_horizon, planning_horizon):
    row_dir = osp.join(out_dir, 'Overlay_Detection_Motion_Planning')
    renderer = BEVRender(ROW_CHOICES['Detection + Motion + Planning'], osp.join(row_dir, 'overlay'),
                         motion_horizon=motion_horizon, planning_horizon=planning_horizon)
    overlay_path = renderer.render_det_motion_planning_overlay(data, base_res, ours_res, idx)
    return [read_and_resize(overlay_path, width)]


def render_pair_gt_overlay_row(data, base_res, ours_res, idx, out_dir, width, motion_horizon, planning_horizon):
    row_dir = osp.join(out_dir, 'Pair_GT_Overlay_Detection_Motion_Planning')
    renderer = BEVRender(ROW_CHOICES['Detection + Motion + Planning'], osp.join(row_dir, 'pair_overlay'),
                         motion_horizon=motion_horizon, planning_horizon=planning_horizon)
    base_path = renderer.render_det_motion_planning_pair_overlay(
        data, base_res, idx, name='Baseline', pred_color=np.asarray([0.12, 0.35, 1.0]))
    ours_path = renderer.render_det_motion_planning_pair_overlay(
        data, ours_res, idx, name='Ours', pred_color=np.asarray([0.95, 0.10, 0.10]))
    return [read_and_resize(base_path, width), read_and_resize(ours_path, width)]



def _occ_points_colors(occ, xlim=40.0, ylim=40.0, zlim=(-1.0, 5.4), ignore_labels=(17, 255)):
    occ = np.asarray(occ)
    h, w, d = occ.shape
    mask = np.ones_like(occ, dtype=bool)
    for label in ignore_labels:
        mask &= occ != label
    xs, ys, zs = np.where(mask)
    labels = occ[xs, ys, zs].astype(np.int64)
    z_min, z_max = zlim
    points = np.stack([
        (xs.astype(np.float32) + 0.5) / h * (2 * xlim) - xlim,
        (ys.astype(np.float32) + 0.5) / w * (2 * ylim) - ylim,
        z_min + (zs.astype(np.float32) + 0.5) / d * (z_max - z_min),
    ], axis=1)
    colors = np.asarray([
        [0, 0, 0], [255, 120, 50], [255, 192, 203], [255, 255, 0],
        [0, 150, 245], [0, 255, 255], [200, 180, 0], [255, 0, 0],
        [255, 240, 150], [135, 60, 0], [160, 32, 240], [255, 0, 255],
        [175, 0, 75], [75, 0, 75], [150, 240, 80], [230, 230, 250],
        [0, 175, 0], [255, 255, 255]], dtype=np.uint8)
    colors = colors[np.clip(labels, 0, len(colors) - 1)]
    return points, colors


def _ego_car_points_colors(voxel_size=(0.2, 0.2, 0.2)):
    # STCOCC-style ego occupancy: a black voxelized car centered at the origin.
    ego_range = [-1.0, -1.0, 0.0, 3.0, 1.0, 1.5]
    vx, vy, vz = voxel_size
    xs = np.arange(ego_range[0] + vx / 2, ego_range[3], vx)
    ys = np.arange(ego_range[1] + vy / 2, ego_range[4], vy)
    zs = np.arange(ego_range[2] + vz / 2, ego_range[5], vz)
    points = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    return points, colors


def _write_voxel_mesh_ply(path, points, colors, voxel_size, add_ego=True):
    dx, dy, dz = voxel_size
    if add_ego:
        ego_points, ego_colors = _ego_car_points_colors((min(dx, 0.2), min(dy, 0.2), min(dz, 0.2)))
        points = np.concatenate([points, ego_points], axis=0)
        colors = np.concatenate([colors, ego_colors], axis=0)
    offsets = np.asarray([
        [-dx/2, -dy/2, -dz/2], [dx/2, -dy/2, -dz/2], [dx/2, dy/2, -dz/2], [-dx/2, dy/2, -dz/2],
        [-dx/2, -dy/2, dz/2], [dx/2, -dy/2, dz/2], [dx/2, dy/2, dz/2], [-dx/2, dy/2, dz/2],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                        [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]], dtype=np.int64)
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {points.shape[0] * 8}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write(f'element face {points.shape[0] * 6}\n')
        f.write('property list uchar int vertex_indices\nend_header\n')
        for p, c in zip(points, colors):
            for v in p[None, :] + offsets:
                f.write(f'{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n')
        for i in range(points.shape[0]):
            base = i * 8
            for face in faces + base:
                f.write('4 ' + ' '.join(map(str, face.tolist())) + '\n')


def _occ_voxel_size_from_cfg(cfg, occ_shape):
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


def export_occ_files(data, base_res, ours_res, idx, out_dir, cfg, export_ply=False):
    token = data.get('token', str(idx))
    occ_dir = osp.join(out_dir, 'exported_occ', f'{idx:04d}_{token}')
    os.makedirs(occ_dir, exist_ok=True)
    gt_occ = np.load(data['ann_info']['occ_path'])['semantics']
    occ_items = {
        'gt': gt_occ,
        'baseline': np.asarray(base_res['pts_occ']['occ']),
        'ours': np.asarray(ours_res['pts_occ']['occ']),
    }
    for name, occ in occ_items.items():
        np.savez_compressed(osp.join(occ_dir, f'{name}.npz'), semantics=occ.astype(np.uint8))
        if export_ply:
            points, colors = _occ_points_colors(occ)
            voxel_size = _occ_voxel_size_from_cfg(cfg, occ.shape)
            _write_voxel_mesh_ply(osp.join(occ_dir, f'{name}_voxels.ply'), points, colors, voxel_size)
    print('exported occ to', occ_dir)



def compose_mixed_rows(row_imgs, out_path):
    rows = []
    for row_name, imgs in row_imgs:
        if len(imgs) == 1:
            row = add_title(imgs[0], row_name)
        else:
            titles = COL_TITLES if len(imgs) == 3 else ['Baseline vs GT', 'Ours vs GT']
            titled = [add_title(img, title) for img, title in zip(imgs, titles)]
            titled = pad_to_same_height(titled)
            row = add_title(cv2.hconcat(titled), row_name, height=46)
        rows.append(row)
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.ones((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8) * 255
            r = np.hstack([r, pad])
        padded_rows.append(r)
    cv2.imwrite(out_path, cv2.vconcat(padded_rows))



def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    baseline = mmcv.load(args.baseline)
    ours = mmcv.load(args.ours)

    for idx in tqdm(args.indices):
        data = dataset.get_data_info(idx)
        if args.export_occ:
            export_occ_files(data, baseline[idx], ours[idx], idx, args.out_dir, cfg, args.export_occ_ply)
        row_imgs = []
        if args.pair_gt_overlay:
            imgs = render_pair_gt_overlay_row(data, baseline[idx], ours[idx], idx, args.out_dir, args.resize_width,
                                              args.motion_horizon, args.planning_horizon)
            row_imgs.append(('Pairwise GT Overlay: prediction color vs dashed GT green', imgs))
        elif args.overlay_det_motion_planning or args.overlay_only:
            imgs = render_overlay_row(data, baseline[idx], ours[idx], idx, args.out_dir, args.resize_width,
                                      args.motion_horizon, args.planning_horizon)
            row_imgs.append(('Overlay: Baseline (blue) / Ours (red) / GT (green)', imgs))
        if not args.overlay_only:
            for row_name, choices in ROW_CHOICES.items():
                occ_kwargs = dict(
                    occ_view=args.occ_view if row_name == 'Occupancy' else 'bev',
                    occ_elev=args.occ_elev,
                    occ_azim=args.occ_azim,
                    occ_marker_size=args.occ_marker_size,
                    occ_alpha=args.occ_alpha,
                    occ_max_points=args.occ_max_points,
                    occ_open3d_view_json=args.occ_open3d_view_json,
                    occ_open3d_use_car=not args.no_occ_open3d_car,
                    occ_open3d_wait_time=args.occ_open3d_wait_time,
                    occ_focus_range=args.occ_focus_range,
                )
                imgs = render_row(row_name, choices, data, baseline[idx], ours[idx], idx, args.out_dir,
                                  args.resize_width, args.motion_horizon, args.planning_horizon, occ_kwargs)
                row_imgs.append((row_name, imgs))
        token = data.get('token', str(idx))
        suffix = 'pair_gt_overlay_grid' if args.pair_gt_overlay else ('overlay_grid' if (args.overlay_det_motion_planning or args.overlay_only) else 'fullstack_grid')
        out_path = osp.join(args.out_dir, f'{idx:04d}_{token}_{suffix}.jpg')
        compose_mixed_rows(row_imgs, out_path)
        print('saved', out_path)


if __name__ == '__main__':
    main()
