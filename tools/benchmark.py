# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
import sys
sys.path.append('.')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from mmdet.models import build_detector
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods
from mmcv.parallel import scatter


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--samples', type=int, default=1000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--warmup', type=int, default=20, help='warmup samples ignored for FPS')
    parser.add_argument(
        '--skip-flops', action='store_true',
        help='skip FLOPs counting (recommended for sparse/Mamba custom ops)')
    parser.add_argument(
        '--skip-speed', action='store_true',
        help='skip FPS/memory benchmark')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def get_max_memory(model):
    device = getattr(model, 'output_device', None)
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([mem / (1024 * 1024)],
        dtype=torch.int,
        device=device)
    return mem_mb.item()


def main():
    args = parse_args()
    if not args.skip_flops:
        try:
            get_flops_params(args)
        except Exception as exc:
            print(f'Original FLOPs counter failed: {repr(exc)}')
            print('Falling back to supported-operator FLOPs counting. '
                  'Sparse/Mamba/custom CUDA ops may be under-counted.')
            get_supported_flops_params(args)
    if not args.skip_speed:
        get_mem_fps(args)

def get_mem_fps(args):
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print(cfg.data.test)
    dataset = custom_build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_module(model)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'Params: {num_params:.2f} M, Trainable params: {num_trainable:.2f} M')

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be slow so skip them
    num_warmup = args.warmup
    pure_inf_time = 0
    timed_samples = 0

    # benchmark with several samples and take the average
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

        max_memory = max(max_memory, get_max_memory(model))
        if i >= num_warmup:
            pure_inf_time += elapsed
            timed_samples += 1
            if timed_samples % args.log_interval == 0:
                fps = timed_samples / pure_inf_time
                latency = pure_inf_time / timed_samples * 1000
                print(f'Done samples [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.2f} samples/s, '
                      f'latency: {latency:.2f} ms, '
                      f'gpu mem: {max_memory / 1024:.2f} GB')

        if (i + 1) == args.samples:
            fps = timed_samples / pure_inf_time if timed_samples > 0 else 0
            latency = pure_inf_time / timed_samples * 1000 if timed_samples > 0 else 0
            print(f'Overall fps: {fps:.2f} samples/s')
            print(f'Overall latency: {latency:.2f} ms/sample')
            print(f'Peak GPU memory: {max_memory / 1024:.2f} GB')
            break



def get_supported_flops_params(args):
    """Best-effort FLOPs counter for the whole detector.

    This uses MMCV's hook-based counter and therefore only counts operators
    supported by MMCV. Sparse convolution, scatter, Mamba kernels, BEV pooling,
    and other custom CUDA ops may be missed or under-counted. Use the result as
    an approximate supported-op FLOPs number, not an exact end-to-end FLOPs.
    """
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    dataset = custom_build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    data = next(iter(dataloader))
    data = scatter(data, [0])[0]

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda(0).eval()

    total_params = sum(p.numel() for p in model.parameters())
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()
    with torch.no_grad():
        flops_model(return_loss=False, rescale=True, **data)
    flops_count, params_count = flops_model.compute_average_flops_cost()
    flops_count *= flops_model.__batch_counter__
    flops_model.stop_flops_count()
    print('Approximate supported-op complexity: '
          f'FLOPs={flops_count / 1e9:.4f} G, '
          f'Params={total_params / 1e6:.4f} M '
          f'(counter params={params_count / 1e6:.4f} M).')


def get_flops_params(args):
    gpu_id = 0
    cfg = Config.fromfile(args.config)
    dataset = custom_build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    data_iter = dataloader.__iter__()
    data = next(data_iter)
    data = scatter(data, [gpu_id])[0]

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda(gpu_id)
    model.eval()

    bilinear_flops = 11
    num_key_pts_det = (
        cfg.model["head"]['det_head']["deformable_model"]["kps_generator"]["num_learnable_pts"]
        + len(cfg.model["head"]['det_head']["deformable_model"]["kps_generator"]["fix_scale"])
    )
    deformable_agg_flops_det = (
        cfg.num_decoder
        * cfg.embed_dims
        * cfg.num_levels
        * cfg.model["head"]['det_head']["instance_bank"]["num_anchor"]
        * cfg.model["head"]['det_head']["deformable_model"]["num_cams"]
        * num_key_pts_det
        * bilinear_flops
    )
    num_key_pts_map = (
        cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["num_learnable_pts"]
        + len(cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["fix_height"])
    ) * cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["num_sample"]
    deformable_agg_flops_map = (
        cfg.num_decoder
        * cfg.embed_dims
        * cfg.num_levels
        * cfg.model["head"]['map_head']["instance_bank"]["num_anchor"]
        * cfg.model["head"]['map_head']["deformable_model"]["num_cams"]
        * num_key_pts_map
        * bilinear_flops
    )
    deformable_agg_flops = deformable_agg_flops_det + deformable_agg_flops_map

    for module in ["total", "img_backbone", "img_neck", "head"]:
        if module != "total":
            flops_model = add_flops_counting_methods(getattr(model, module))
        else:
            flops_model = add_flops_counting_methods(model)
        flops_model.eval()
        flops_model.start_flops_count()

        if module == "img_backbone":
            flops_model(data["img"].flatten(0, 1))
        elif module == "img_neck":
            flops_model(model.img_backbone(data["img"].flatten(0, 1)))
        elif module == "head":
            flops_model(model.extract_feat(data["img"], metas=data), data)
        else:
            flops_model(**data)
        flops_count, params_count = flops_model.compute_average_flops_cost()
        flops_count *= flops_model.__batch_counter__
        flops_model.stop_flops_count()
        if module == "head" or module == "total":
            flops_count += deformable_agg_flops
        if module == "total":
            total_flops = flops_count
            total_params = params_count
        print(
            f"{module:<13} complexity: "
            f"FLOPs={flops_count/ 10.**9:>8.4f} G / {flops_count/total_flops*100:>6.2f}%, "
            f"Params={params_count/10**6:>8.4f} M / {params_count/total_params*100:>6.2f}%."
        )

if __name__ == '__main__':
    main()
