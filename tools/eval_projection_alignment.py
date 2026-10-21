import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def coords_to_keys(coords, shape):
    # coords: [N, 3] in z,y,x order.
    return coords[:, 0] * (shape[1] * shape[2]) + coords[:, 1] * shape[2] + coords[:, 2]


def occupancy_grid(coords, shape, device):
    grid = torch.zeros(int(shape[0] * shape[1] * shape[2]), dtype=torch.bool, device=device)
    if coords.numel() > 0:
        mask = ((coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
                (coords[:, 1] >= 0) & (coords[:, 1] < shape[1]) &
                (coords[:, 2] >= 0) & (coords[:, 2] < shape[2]))
        coords = coords[mask].long()
        if coords.numel() > 0:
            grid[coords_to_keys(coords, shape)] = True
    return grid.view(1, 1, shape[0], shape[1], shape[2])


def mean_chamfer_voxel(cam_coords, lidar_coords, max_pairs=200000):
    if cam_coords.numel() == 0 or lidar_coords.numel() == 0:
        return torch.tensor(0., device=cam_coords.device), torch.tensor(0., device=cam_coords.device)
    if cam_coords.shape[0] > max_pairs:
        cam_coords = cam_coords[torch.linspace(0, cam_coords.shape[0] - 1, max_pairs, device=cam_coords.device).long()]
    if lidar_coords.shape[0] > max_pairs:
        lidar_coords = lidar_coords[torch.linspace(0, lidar_coords.shape[0] - 1, max_pairs, device=lidar_coords.device).long()]
    d = torch.cdist(cam_coords.float(), lidar_coords.float())
    return d.min(dim=1).values.sum(), torch.tensor(cam_coords.shape[0], dtype=torch.float32, device=cam_coords.device)


def occ_mask(coords, fg, grid_shape):
    # Map UniLION z/y/x indices to Occ3D grid indices before foreground lookup.
    occ_shape = torch.tensor(fg.shape, device=coords.device, dtype=torch.float32)
    src_shape = torch.tensor(grid_shape, device=coords.device, dtype=torch.float32)
    occ = torch.floor(coords.float() * occ_shape.view(1, 3) / src_shape.view(1, 3)).long()
    valid = ((occ[:, 0] >= 0) & (occ[:, 0] < fg.shape[0]) &
             (occ[:, 1] >= 0) & (occ[:, 1] < fg.shape[1]) &
             (occ[:, 2] >= 0) & (occ[:, 2] < fg.shape[2]))
    out = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
    if valid.any():
        v = occ[valid]
        out[valid] = fg[v[:, 0], v[:, 1], v[:, 2]]
    return out


def compute_metrics(cam_idx, lidar_idx, gt_occ, shape, free_id=17):
    device = cam_idx.device
    stats = torch.zeros(5, dtype=torch.float64, device=device)
    gt_occ = gt_occ.to(device)
    if gt_occ.dim() == 4:
        gt_occ = gt_occ[0]
    fg = (gt_occ != free_id) & (gt_occ != 255)

    for b in torch.unique(cam_idx[:, 0]).tolist() if cam_idx.numel() else []:
        cam = cam_idx[cam_idx[:, 0] == b][:, 1:].long()
        lid = lidar_idx[lidar_idx[:, 0] == b][:, 1:].long()
        if cam.numel() == 0 or lid.numel() == 0:
            continue
        cam = cam[((cam[:, 0] >= 0) & (cam[:, 0] < shape[0]) &
                   (cam[:, 1] >= 0) & (cam[:, 1] < shape[1]) &
                   (cam[:, 2] >= 0) & (cam[:, 2] < shape[2]))]
        lid = lid[((lid[:, 0] >= 0) & (lid[:, 0] < shape[0]) &
                  (lid[:, 1] >= 0) & (lid[:, 1] < shape[1]) &
                  (lid[:, 2] >= 0) & (lid[:, 2] < shape[2]))]
        cam = cam[occ_mask(cam, fg, shape)]
        lid = lid[occ_mask(lid, fg, shape)]
        if cam.numel() == 0 or lid.numel() == 0:
            continue
        lidar_grid = occupancy_grid(lid, shape, device)
        cam_grid = occupancy_grid(cam, shape, device)
        cam_count = cam_grid.sum().double().clamp_min(1.)
        overlap1 = (cam_grid & lidar_grid).sum().double()
        dilated = F.max_pool3d(lidar_grid.float(), kernel_size=3, stride=1, padding=1).bool()
        overlap3 = (cam_grid & dilated).sum().double()
        dist_sum, dist_count = mean_chamfer_voxel(cam, lid)
        stats[0] += dist_sum.double()
        stats[1] += dist_count.double()
        stats[2] += overlap1
        stats[3] += overlap3
        stats[4] += cam_count
    return stats



def scale_lidar_to_img_grid(lidar_idx, lidar_shape, img_shape):
    out = lidar_idx.clone().long()
    for i in range(3):
        if lidar_shape[i] != img_shape[i]:
            out[:, i + 1] = torch.floor(out[:, i + 1].float() * img_shape[i] / lidar_shape[i]).long()
    return out


def compute_batch(model, batch, lidar_shape, img_shape):
    store = {}
    core = model.module if hasattr(model, 'module') else model

    def hook_img(_, __, out):
        store['cam_idx'] = out[1].detach()

    def hook_pts(_, __, out):
        store['lidar_idx'] = out[1].detach()

    h1 = core.img_map2bev.register_forward_hook(hook_img)
    h2 = core.pts_voxel_encoder.register_forward_hook(hook_pts)
    with torch.no_grad():
        _ = model(return_loss=False, rescale=True, **batch)
    h1.remove(); h2.remove()
    gt_occ = unwrap(batch['gt_occ'])
    lidar_idx = scale_lidar_to_img_grid(store['lidar_idx'], lidar_shape, img_shape).to(store['cam_idx'].device)
    return compute_metrics(store['cam_idx'], lidar_idx, gt_occ, img_shape)


def main():
    parser = argparse.ArgumentParser(description='Evaluate camera-lifted voxel vs LiDAR voxel alignment.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out', default=None)
    parser.add_argument('--samples-per-gpu', type=int, default=1)
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
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
            print('WARNING: forcing samples_per_gpu=1 for variable-size fields.')
        args.samples_per_gpu = 1

    collect = cfg.data.val.pipeline[-1]['transforms'][-1]
    if 'gt_occ' not in collect['keys']:
        collect['keys'].append('gt_occ')

    cfg.data.val.work_dir = cfg.get('work_dir', './work_dirs/projection_alignment')
    dataset = build_dataset(cfg.data.val)
    if distributed:
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=args.workers_per_gpu,
                                  dist=True, shuffle=False, nonshuffler_sampler=dict(type='DistributedSampler'))
    else:
        loader = build_dataloader_origin(dataset, samples_per_gpu=1,
                                         workers_per_gpu=args.workers_per_gpu, dist=False, shuffle=False)
    model = load_model(cfg, args.checkpoint, distributed=distributed)
    img_shape = tuple(int(x) for x in cfg.model.img_map2bev.grid_config['grid_size']) if False else tuple(int(x) for x in cfg.grid_size[::-1])
    lidar_shape = tuple(int(x) for x in cfg.grid_size[::-1])

    stats = torch.zeros(5, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
    iterator = tqdm(loader, total=len(loader)) if rank == 0 else loader
    for i, batch in enumerate(iterator):
        if args.max_samples is not None and i >= args.max_samples:
            break
        stats += compute_batch(model, batch, lidar_shape, img_shape)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if rank != 0:
        return
    denom_dist = max(stats[1].item(), 1.0)
    denom_cam = max(stats[4].item(), 1.0)
    result = dict(
        foreground_dist=float(stats[0].item() / denom_dist),
        overlap1=float(stats[2].item() / denom_cam),
        overlap3=float(stats[3].item() / denom_cam),
        camera_fg_voxels=int(stats[4].item()),
        dist_samples=int(stats[1].item()),
        world_size=world_size,
        config=args.config,
        checkpoint=args.checkpoint)
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
