#!/usr/bin/env python3
import argparse
import csv
import os
import pickle

import numpy as np
import torch


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, 'gravity_center'):
        return x.gravity_center.detach().cpu().numpy()
    if hasattr(x, 'tensor'):
        x = x.tensor
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pred_centers(result, score_thr=0.3):
    bbox = result.get('pts_bbox', {})
    boxes = bbox.get('boxes_3d', None)
    scores = to_numpy(bbox.get('scores_3d', None))
    labels = to_numpy(bbox.get('labels_3d', None))
    if boxes is None:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)
    centers = to_numpy(boxes)[:, :2]
    if scores is None:
        keep = np.ones((len(centers),), dtype=bool)
        scores = np.ones((len(centers),), dtype=np.float32)
    else:
        keep = scores >= score_thr
    if labels is None:
        labels = np.zeros((len(centers),), dtype=np.int64)
    return centers[keep], scores[keep], labels[keep]


def gt_centers_labels(info):
    gt_boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))))
    centers = gt_boxes[:, :2] if len(gt_boxes) else np.zeros((0, 2))
    labels = info.get('gt_labels_3d', info.get('gt_labels', None))
    if labels is None:
        labels = np.zeros((len(centers),), dtype=np.int64)
        class_aware = False
    else:
        labels = np.asarray(labels).astype(np.int64)
        class_aware = len(labels) == len(centers)
    return centers, labels, class_aware


def det_metrics(result, info, score_thr=0.3, dist_thr=2.0):
    gt_centers, gt_labels, class_aware = gt_centers_labels(info)
    centers, scores, labels = pred_centers(result, score_thr)
    num_gt, num_pred = len(gt_centers), len(centers)
    if num_gt == 0:
        return dict(recall=0.0, precision=0.0, fp=num_pred, mean_min_dist=99.0, num_pred=num_pred, num_gt=0)
    if num_pred == 0:
        return dict(recall=0.0, precision=0.0, fp=0, mean_min_dist=99.0, num_pred=0, num_gt=num_gt)
    dist = np.linalg.norm(gt_centers[:, None, :] - centers[None, :, :], axis=-1)
    valid = dist < dist_thr
    if class_aware and len(labels) == num_pred:
        valid &= gt_labels[:, None] == labels[None, :]
    matched_gt, tp = set(), 0
    for pi in np.argsort(scores)[::-1]:
        cand = np.where(valid[:, pi])[0]
        cand = [g for g in cand if g not in matched_gt]
        if cand:
            gi = min(cand, key=lambda g: dist[g, pi])
            matched_gt.add(int(gi)); tp += 1
    masked_dist = np.where(valid, dist, np.inf)
    min_dist = np.min(masked_dist, axis=1)
    min_dist[~np.isfinite(min_dist)] = 99.0
    fp = num_pred - tp
    return dict(recall=float(tp / num_gt), precision=float(tp / max(num_pred, 1)), fp=int(fp),
                mean_min_dist=float(min_dist.mean()), num_pred=int(num_pred), num_gt=int(num_gt))


def plan_l2(result, info):
    plan = result.get('pts_plan_reg', {}).get('plan_reg', None)
    if plan is None or 'gt_ego_fut_trajs' not in info:
        return 99.0
    plan = to_numpy(plan)
    gt = np.cumsum(np.asarray(info['gt_ego_fut_trajs']), axis=0)
    mask = np.asarray(info.get('gt_ego_fut_masks', np.ones((gt.shape[0],)))).astype(bool)
    t = min(len(plan), len(gt), len(mask))
    if t == 0 or not mask[:t].any():
        return 99.0
    l2 = np.linalg.norm(plan[:t] - gt[:t], axis=-1)
    return float(l2[mask[:t]].mean())


def norm_plan_gain(b_plan, o_plan):
    if b_plan >= 50 or o_plan >= 50:
        return 0.0
    return float(np.clip((b_plan - o_plan) / max(b_plan, 1.0), -1.0, 1.0))


def occ_iou(result, info):
    occ_path = info.get('occ_path', None)
    pred = result.get('pts_occ', {}).get('occ', None)
    if occ_path is None or pred is None:
        return None
    if not os.path.isabs(occ_path) and not os.path.exists(occ_path):
        occ_path = os.path.join('data/nuscenes', occ_path)
    try:
        gt = np.load(occ_path)['semantics']
    except Exception:
        return None
    pred = np.asarray(pred)
    pred_bev = (pred > 0).max(axis=-1) if pred.ndim == 3 else pred > 0
    gt_bev = (gt != 17).max(axis=-1) if gt.ndim == 3 else gt != 17
    h, w = min(pred_bev.shape[0], gt_bev.shape[0]), min(pred_bev.shape[1], gt_bev.shape[1])
    inter = np.logical_and(pred_bev[:h, :w], gt_bev[:h, :w]).sum()
    union = np.logical_or(pred_bev[:h, :w], gt_bev[:h, :w]).sum()
    return float(inter / max(union, 1))


def clarity_score(num_gt, num_pred):
    gt_score = np.clip(1.0 - abs(num_gt - 25) / 25.0, 0.0, 1.0)
    pred_score = 1.0 if num_pred >= 4 else 0.3
    return float(0.7 * gt_score + 0.3 * pred_score)


def select_diverse(rows, topk, min_gap=20, max_per_scene=1):
    selected, scene_count = [], {}
    for r in rows:
        if any(abs(r['index'] - s['index']) < min_gap for s in selected):
            continue
        scene = r.get('scene_token', '')
        if scene and max_per_scene > 0 and scene_count.get(scene, 0) >= max_per_scene:
            continue
        selected.append(r); scene_count[scene] = scene_count.get(scene, 0) + 1
        if len(selected) >= topk:
            break
    return selected


def main():
    parser = argparse.ArgumentParser('Select qualitative samples for full-stack visualization.')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--ours', required=True)
    parser.add_argument('--info', default='data/nuscenes/nuscenes_infos_val.pkl')
    parser.add_argument('--out-dir', default='vis/qualitative_selection')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--mode', choices=['fullstack', 'planning_det'], default='fullstack')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--det-dist-thr', type=float, default=2.0)
    parser.add_argument('--min-gt-boxes', type=int, default=5)
    parser.add_argument('--max-gt-boxes', type=int, default=70)
    parser.add_argument('--no-diverse', action='store_true')
    parser.add_argument('--min-index-gap', type=int, default=20)
    parser.add_argument('--max-per-scene', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    baseline, ours = load_pkl(args.baseline), load_pkl(args.ours)
    infos = load_pkl(args.info)['infos']
    assert len(baseline) == len(ours) == len(infos)
    rows = []
    for i, info in enumerate(infos):
        num_gt = len(info.get('gt_boxes', []))
        if num_gt < args.min_gt_boxes or num_gt > args.max_gt_boxes:
            continue
        b_plan, o_plan = plan_l2(baseline[i], info), plan_l2(ours[i], info)
        b_det = det_metrics(baseline[i], info, args.score_thr, args.det_dist_thr)
        o_det = det_metrics(ours[i], info, args.score_thr, args.det_dist_thr)
        b_occ, o_occ = occ_iou(baseline[i], info), occ_iou(ours[i], info)
        occ_gain = 0.0 if b_occ is None or o_occ is None else o_occ - b_occ
        plan_gain, plan_gain_norm = b_plan - o_plan, norm_plan_gain(b_plan, o_plan)
        det_recall_gain = o_det['recall'] - b_det['recall']
        det_precision_gain = o_det['precision'] - b_det['precision']
        det_dist_gain = b_det['mean_min_dist'] - o_det['mean_min_dist']
        det_dist_gain_norm = float(np.clip(det_dist_gain / 10.0, -1.0, 1.0))
        clarity = clarity_score(num_gt, o_det['num_pred'])
        penalty = 0.0
        penalty += float(det_recall_gain < -0.15)
        penalty += float(det_precision_gain < -0.20)
        penalty += float(o_plan > b_plan + 1.0 and b_plan < 50)
        penalty += float(occ_gain < -0.03)
        if args.mode == 'planning_det':
            score = 2.0 * plan_gain + det_recall_gain + 0.2 * det_dist_gain
        else:
            score = (1.5 * occ_gain + plan_gain_norm + 0.8 * det_recall_gain +
                     0.3 * det_precision_gain + 0.3 * det_dist_gain_norm + 0.4 * clarity - 0.8 * penalty)
        rows.append(dict(index=i, token=info.get('token', ''), scene_token=info.get('scene_token', ''),
                         score=score, mode=args.mode, baseline_plan_l2=b_plan, ours_plan_l2=o_plan,
                         plan_gain=plan_gain, plan_gain_norm=plan_gain_norm, baseline_occ_iou=b_occ,
                         ours_occ_iou=o_occ, occ_gain=occ_gain, baseline_det_recall=b_det['recall'],
                         ours_det_recall=o_det['recall'], det_recall_gain=det_recall_gain,
                         baseline_det_precision=b_det['precision'], ours_det_precision=o_det['precision'],
                         det_precision_gain=det_precision_gain, baseline_det_dist=b_det['mean_min_dist'],
                         ours_det_dist=o_det['mean_min_dist'], det_dist_gain=det_dist_gain,
                         clarity=clarity, degradation_penalty=penalty, baseline_num_pred=b_det['num_pred'],
                         ours_num_pred=o_det['num_pred'], num_gt=b_det['num_gt']))
    rows = sorted(rows, key=lambda x: x['score'], reverse=True)
    if not rows:
        raise RuntimeError('No candidate rows found. Check inputs and filtering thresholds.')
    out_csv = os.path.join(args.out_dir, 'qualitative_candidates.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    top = rows[:args.topk] if args.no_diverse else select_diverse(rows, args.topk, args.min_index_gap, args.max_per_scene)
    out_txt = os.path.join(args.out_dir, 'top_indices.txt')
    with open(out_txt, 'w') as f:
        f.write(' '.join(str(r['index']) for r in top) + '\n')
    print('Saved:', out_csv); print('Saved:', out_txt)
    print('Top indices:', ' '.join(str(r['index']) for r in top[:20]))
    for r in top[:10]:
        print(r)


if __name__ == '__main__':
    main()
