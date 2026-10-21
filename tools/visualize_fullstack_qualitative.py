#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon, Rectangle

try:
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
except Exception:
    LiDARInstance3DBoxes = None




def infer_dist_order(num_samples, world_size):
    order = []
    for rank in range(world_size):
        order.extend(list(range(rank, num_samples, world_size)))
    return np.asarray(order[:num_samples], dtype=np.int64)


def reorder_results_if_needed(results, order):
    if order is None:
        return results
    restored = [None] * len(order)
    for gathered_idx, dataset_idx in enumerate(order):
        restored[int(dataset_idx)] = results[gathered_idx]
    return restored

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, 'tensor'):
        x = x.tensor
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def box_corners_bev(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def bev_corners_from_boxes(boxes):
    # mmdet3d's 8 corners are not ordered as a BEV quadrilateral in the
    # first four entries.  For the default corner order, [0, 3, 7, 4]
    # forms one horizontal face and gives the four distinct BEV corners.
    bev_idx = [0, 3, 7, 4]
    if hasattr(boxes, 'corners'):
        corners = boxes.corners.detach().cpu().numpy()
        return corners[:, bev_idx, :2]
    boxes_np = to_numpy(boxes)
    if boxes_np is None or len(boxes_np) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    if LiDARInstance3DBoxes is not None:
        try:
            b = LiDARInstance3DBoxes(torch.as_tensor(boxes_np[:, :7]).float(), box_dim=7)
            corners = b.corners.detach().cpu().numpy()
            return corners[:, bev_idx, :2]
        except Exception:
            pass
    return np.stack([box_corners_bev(b) for b in boxes_np], axis=0)


def draw_boxes(ax, boxes, scores=None, labels=None, color='tab:blue', score_thr=0.3, max_boxes=80, lw=1.0):
    scores = to_numpy(scores)
    corners = bev_corners_from_boxes(boxes)
    if corners is None or len(corners) == 0:
        return
    order = np.arange(len(corners))
    if scores is not None:
        order = order[scores >= score_thr]
        order = order[np.argsort(scores[order])[::-1]]
    for idx in order[:max_boxes]:
        poly = Polygon(corners[idx], closed=True, fill=False, edgecolor=color, linewidth=lw, alpha=0.9)
        ax.add_patch(poly)




def orient_raster_for_bev(bev, mode='rot90cw'):
    if mode == 'none':
        return bev
    if mode == 'transpose':
        return bev.T
    if mode == 'rot90cw':
        return np.rot90(bev, k=-1)
    if mode == 'rot90ccw':
        return np.rot90(bev, k=1)
    if mode == 'flipx':
        return np.flip(bev, axis=0)
    if mode == 'flipy':
        return np.flip(bev, axis=1)
    raise ValueError(f'Unknown raster orientation mode: {mode}')

def draw_occ(ax, occ, extent, alpha=0.28, raster_mode='rot90cw'):
    occ = to_numpy(occ)
    if occ is None:
        return
    if occ.ndim == 3:
        bev = (occ > 0).max(axis=-1).astype(float)
    elif occ.ndim == 2:
        bev = (occ > 0).astype(float)
    else:
        return
    bev = orient_raster_for_bev(bev, raster_mode)
    ax.imshow(bev, origin='lower', extent=extent, cmap='Oranges', alpha=alpha, interpolation='nearest')


def draw_seg(ax, seg, extent, alpha=0.18, raster_mode='rot90cw'):
    seg = to_numpy(seg)
    if seg is None:
        return
    if seg.ndim != 3:
        return
    colors = ['Greens', 'Purples', 'Blues', 'Greys', 'pink', 'winter']
    # Show drivable_area, ped_crossing, walkway, divider if available.
    for ch in [0, 1, 2, 5]:
        if ch >= seg.shape[0]:
            continue
        m = seg[ch]
        if torch.is_tensor(m):
            m = m.detach().cpu().numpy()
        if m.max() > 1 or m.min() < 0:
            m = 1 / (1 + np.exp(-m))
        m = orient_raster_for_bev(m, raster_mode)
        masked = np.ma.masked_less(m, 0.35)
        ax.imshow(masked, origin='lower', extent=extent, cmap=colors[ch % len(colors)], alpha=alpha, interpolation='nearest')


def draw_motion(ax, result, color='tab:purple', score_thr=0.35, max_agents=25):
    bbox = result.get('pts_bbox', {})
    boxes = to_numpy(bbox.get('boxes_3d'))
    scores = to_numpy(bbox.get('scores_3d'))
    trajs = to_numpy(bbox.get('trajs_3d'))
    trajs_cls = to_numpy(bbox.get('trajs_cls_3d'))
    if boxes is None or trajs is None or trajs_cls is None:
        return
    keep = np.arange(len(boxes))
    if scores is not None:
        keep = keep[scores >= score_thr]
        keep = keep[np.argsort(scores[keep])[::-1]]
    for idx in keep[:max_agents]:
        mode = int(np.argmax(trajs_cls[idx]))
        tr = trajs[idx, mode]
        # Agent trajectories are offsets in many UniLION outputs; draw from box center.
        if np.linalg.norm(tr[0]) < 5.0:
            tr = np.cumsum(tr, axis=0) + boxes[idx, :2]
        ax.plot(tr[:, 0], tr[:, 1], color=color, linewidth=1.0, alpha=0.65)


def draw_plan(ax, result, color='red', label=None, lw=2.5):
    plan = result.get('pts_plan_reg', {}).get('plan_reg')
    plan = to_numpy(plan)
    if plan is None:
        return
    ax.plot(plan[:, 0], plan[:, 1], color=color, linewidth=lw, label=label)
    ax.scatter(plan[-1, 0], plan[-1, 1], color=color, s=14)


def draw_gt(ax, info, class_names, color='lime'):
    boxes = info.get('gt_boxes', None)
    if boxes is not None and LiDARInstance3DBoxes is not None:
        boxes = LiDARInstance3DBoxes(torch.as_tensor(boxes[:, :7]).float(), box_dim=7, origin=(0.5, 0.5, 0.5))
    draw_boxes(ax, boxes, color=color, score_thr=-1, max_boxes=100, lw=1.2)
    gt_plan = info.get('gt_ego_fut_trajs', None)
    if gt_plan is not None:
        gt_plan = np.cumsum(gt_plan, axis=0)
        ax.plot(gt_plan[:, 0], gt_plan[:, 1], color='lime', linewidth=2.5, linestyle='--', label='GT plan')


def setup_ax(ax, title, pc_range):
    ax.set_title(title, fontsize=11)
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.2, alpha=0.3)
    ax.add_patch(Rectangle((-1.0, -2.0), 2.0, 4.0, facecolor='black', alpha=0.8))


def draw_panel(ax, result, info, title, pc_range, is_gt=False, raster_mode='rot90cw'):
    extent = [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]
    setup_ax(ax, title, pc_range)
    if is_gt:
        draw_gt(ax, info, None)
        return
    draw_occ(ax, result.get('pts_occ', {}).get('occ'), extent, raster_mode=raster_mode)
    draw_seg(ax, result.get('pts_seg_map', {}).get('seg'), extent, raster_mode=raster_mode)
    bbox = result.get('pts_bbox', {})
    draw_boxes(ax, bbox.get('boxes_3d'), bbox.get('scores_3d'), bbox.get('labels_3d'), color='tab:blue')
    draw_motion(ax, result)
    draw_plan(ax, result, color='red')
    draw_gt(ax, info, None, color='lime')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--ours', required=True)
    parser.add_argument('--info', default='data/nuscenes/nuscenes_infos_val.pkl')
    parser.add_argument('--out-dir', default='vis/fullstack')
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    parser.add_argument('--pc-range', nargs=6, type=float, default=[-54, -54, -3, 54, 54, 5])
    parser.add_argument('--raster-mode', default='rot90cw',
                        choices=['none', 'transpose', 'rot90cw', 'rot90ccw', 'flipx', 'flipy'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--restore-order', action='store_true')


    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    baseline = load_pkl(args.baseline)
    ours = load_pkl(args.ours)
    infos = load_pkl(args.info)['infos']
    if args.restore_order:
        order = infer_dist_order(len(infos), args.world_size)
        baseline = reorder_results_if_needed(baseline, order)
        ours = reorder_results_if_needed(ours, order)
    for idx in args.indices:
        if args.debug:
            b_num = len(baseline[idx].get('pts_bbox', {}).get('scores_3d', []))
            o_num = len(ours[idx].get('pts_bbox', {}).get('scores_3d', []))
            gt_num = len(infos[idx].get('gt_boxes', []))
            print(f'idx={idx} token={infos[idx].get("token", "NA")} baseline_boxes={b_num} ours_boxes={o_num} gt_boxes={gt_num}')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
        draw_panel(axes[0], baseline[idx], infos[idx], 'Baseline', args.pc_range, raster_mode=args.raster_mode)
        draw_panel(axes[1], ours[idx], infos[idx], 'Ours', args.pc_range, raster_mode=args.raster_mode)
        draw_panel(axes[2], {}, infos[idx], 'GT / Reference', args.pc_range, is_gt=True, raster_mode=args.raster_mode)
        token = infos[idx].get('token', str(idx))
        fig.tight_layout()
        out = os.path.join(args.out_dir, f'{idx:04d}_{token}.png')
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('saved', out)


if __name__ == '__main__':
    main()
