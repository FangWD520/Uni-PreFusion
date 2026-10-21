#!/usr/bin/env python3
import argparse
import importlib
import os
import sys

import torch

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset


def add_pseudo_instance_ids(results):
    """Tracking formatter expects instance_ids when thresholding.

    UniLION does not train a tracking/association head; this creates stable
    per-sample pseudo IDs so the existing tracking-by-detection formatter can
    run without crashing. Metrics should be interpreted accordingly.
    """
    for sample_idx, item in enumerate(results):
        bbox = item.get('pts_bbox', item)
        if 'instance_ids' in bbox:
            continue
        n = len(bbox['scores_3d'])
        device = bbox['scores_3d'].device if hasattr(bbox['scores_3d'], 'device') else None
        ids = torch.arange(n, dtype=torch.long, device=device) + sample_idx * 100000
        bbox['instance_ids'] = ids



def import_plugins(cfg):
    if cfg.get('plugin', False):
        plugin_dir = cfg.plugin_dir
        module_dir = os.path.dirname(plugin_dir).split('/')
        module_path = module_dir[0]
        for m in module_dir[1:]:
            module_path += '.' + m
        importlib.import_module(module_path)


def parse_args():
    parser = argparse.ArgumentParser('Evaluate nuScenes tracking from saved results.pkl')
    parser.add_argument('config', help='Config used to build the val dataset')
    parser.add_argument('--results', required=True, help='Path to results.pkl')
    parser.add_argument('--out-dir', required=True, help='Directory to save tracking json/metrics')
    parser.add_argument('--tracking-threshold', type=float, default=0.2)
    parser.add_argument('--workers-per-gpu', type=int, default=None,
                        help='Unused; kept for compatibility with other scripts.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugins(cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    # Build val dataset and force tracking outputs to be written to out_dir.
    dataset_cfg = cfg.data.val
    dataset_cfg.work_dir = args.out_dir
    dataset = build_dataset(dataset_cfg)
    dataset.work_dir = args.out_dir

    results = mmcv.load(args.results)
    add_pseudo_instance_ids(results)

    print(f'Loaded {len(results)} results from {args.results}')
    print(f'Dataset size: {len(dataset)}')
    assert len(results) == len(dataset), (
        f'Results length {len(results)} != dataset length {len(dataset)}')

    dataset.tracking = True
    dataset.tracking_threshold = args.tracking_threshold
    result_files, tmp_dir = dataset.format_results(
        results, jsonfile_prefix=args.out_dir, tracking=True)

    metrics = {}
    if isinstance(result_files, dict):
        for name, result_path in result_files.items():
            metrics.update(dataset._evaluate_single(
                result_path, result_name=name, tracking=True))
    else:
        metrics.update(dataset._evaluate_single(result_files, tracking=True))
    if tmp_dir is not None:
        tmp_dir.cleanup()

    print('\nTracking metrics:')
    for k in sorted(metrics.keys()):
        if 'amota' in k or 'amotp' in k or 'mota' in k or 'motp' in k or 'recall' in k or 'ids' in k:
            print(f'{k}: {metrics[k]}')
    print(f'All outputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
