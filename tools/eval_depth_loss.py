import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.datasets import build_dataset, build_dataloader as build_dataloader_origin
from mmdet.models import build_detector
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import projects.mmdet3d_plugin  # noqa: F401
import projects.mmdet3d_plugin.datasets  # noqa: F401
import projects.mmdet3d_plugin.models  # noqa: F401


def unwrap(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        return unwrap(x[0]) if len(x) > 0 else x
    if hasattr(x, 'data'):
        return unwrap(x.data)
    return x


def load_model(cfg, checkpoint, distributed=False):
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.eval()
    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
            model = MMDistributedDataParallel(
                model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        else:
            model = MMDataParallel(model.cuda(), device_ids=[0])
    return model


def depth_expected(depth_prob, depth_cfg):
    dmin, dmax, step = depth_cfg
    bins = torch.arange(dmin, dmax, step, device=depth_prob.device, dtype=depth_prob.dtype)
    bins = bins[:depth_prob.shape[1]]
    return (depth_prob * bins.view(1, -1, 1, 1)).sum(dim=1)


def compute_depth_metrics(model, batch, depth_cfg):
    store = {}
    core = model.module if hasattr(model, 'module') else model

    def hook(_, __, out):
        store['depth'] = out[3]

    handle = core.img_map2bev.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(return_loss=False, rescale=True, **batch)
    handle.remove()

    depth = store['depth']
    gt_depth = unwrap(batch['gt_depth']).to(depth.device)
    bce = core.img_map2bev.get_depth_loss(gt_depth, depth)
    gt_onehot = core.img_map2bev.get_downsampled_gt_depth(gt_depth)
    valid = gt_onehot.max(dim=1).values > 0
    gt_bin = gt_onehot.argmax(dim=1).to(depth.dtype)
    dmin, _, step = depth_cfg
    gt_metric = dmin + gt_bin * step
    pred_metric = depth_expected(depth, depth_cfg)
    pred = pred_metric.reshape(-1)[valid].clamp_min(1e-3)
    gt = gt_metric.reshape(-1)[valid].clamp_min(1e-3)
    count = gt.numel()
    if count == 0:
        return torch.zeros(9, dtype=torch.float64, device=depth.device)
    err = (pred - gt).abs()
    ratio = torch.maximum(pred / gt, gt / pred)
    out = torch.zeros(9, dtype=torch.float64, device=depth.device)
    out[0] = bce.double() * count
    out[1] = count
    out[2] = bce.double()
    out[3] = 1
    out[4] = (err / gt).sum().double()
    out[5] = err.sum().double()
    out[6] = ((pred - gt) ** 2).sum().double()
    out[7] = (ratio < 1.25).sum().double()
    out[8] = (ratio < 1.25 ** 2).sum().double()
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate only LSS depth loss on val set.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out', default=None, help='Optional JSON output path.')
    parser.add_argument('--samples-per-gpu', type=int, default=1)
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None, help='Debug: evaluate first N local batches only.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    cfg = Config.fromfile(args.config)
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()

    if args.samples_per_gpu != 1:
        if rank == 0:
            print('WARNING: forcing --samples-per-gpu 1 because UniLION val samples contain variable-size fields that cannot be stacked safely.')
        args.samples_per_gpu = 1

    cfg.data.val.work_dir = cfg.get('work_dir', './work_dirs/depth_loss')
    dataset = build_dataset(cfg.data.val)
    if distributed:
        loader = build_dataloader(
            dataset,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
            dist=True,
            shuffle=False,
            nonshuffler_sampler=dict(type='DistributedSampler'))
    else:
        loader = build_dataloader_origin(
            dataset, samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu, dist=False, shuffle=False)
    model = load_model(cfg, args.checkpoint, distributed=distributed)

    stats = torch.zeros(9, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
    iterator = tqdm(loader, total=len(loader)) if rank == 0 else loader
    for i, batch in enumerate(iterator):
        if args.max_samples is not None and i >= args.max_samples:
            break
        stats += compute_depth_metrics(model, batch, cfg.grid_config['depth'])

    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if rank != 0:
        return

    n = max(stats[1].item(), 1.0)
    batches = max(stats[3].item(), 1.0)
    result = dict(
        depth_bce=float(stats[0].item() / n),
        depth_bce_batch_mean=float(stats[2].item() / batches),
        abs_rel=float(stats[4].item() / n),
        mae=float(stats[5].item() / n),
        rmse=float((stats[6].item() / n) ** 0.5),
        delta1=float(stats[7].item() / n),
        delta2=float(stats[8].item() / n),
        valid_pixels=int(stats[1].item()),
        batches=int(stats[3].item()),
        samples_per_gpu=args.samples_per_gpu,
        world_size=world_size,
        config=args.config,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result, indent=2))
    if args.out is not None:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
