import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models import build_detector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import projects.mmdet3d_plugin  # noqa: F401
import projects.mmdet3d_plugin.datasets  # noqa: F401
import projects.mmdet3d_plugin.models  # noqa: F401


def load_state(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        for key in ['state_dict', 'model_state', 'model']:
            if key in obj and isinstance(obj[key], dict):
                return obj[key], obj
    return obj, obj


def shape(x):
    return tuple(x.shape) if hasattr(x, 'shape') else None


def import_plugin(cfg):
    if getattr(cfg, 'plugin', False):
        import importlib
        plugin_dir = cfg.plugin_dir
        module_dir = os.path.dirname(plugin_dir).split('/')
        module_path = module_dir[0]
        for m in module_dir[1:]:
            module_path += '.' + m
        importlib.import_module(module_path)


def build_target_state(config):
    cfg = Config.fromfile(config)
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    return model.state_dict()


def copy_metadata(src, dst):
    md = getattr(src, '_metadata', None)
    if md is not None:
        dst._metadata = md.copy()


def main():
    parser = argparse.ArgumentParser(
        description='Merge UniLION/MambaFusion checkpoint with ImageNet ResNet50 weights for ResNet image-backbone ablation.')
    parser.add_argument('--config', default='projects/configs/unilion_resnet50_384_seq_e2e.py')
    parser.add_argument('--base', default='ckpts/unilion_384_seq_perception_from_det_map_init.pth')
    parser.add_argument('--resnet', default='ckpts/resnet50-19c8e357.pth')
    parser.add_argument('--out', default='ckpts/unilion_resnet50_384_seq_perception_from_det_map_init.pth')
    args = parser.parse_args()

    target_sd = build_target_state(args.config)
    base_sd, base_obj = load_state(args.base)
    resnet_sd, _ = load_state(args.resnet)

    merged = OrderedDict((k, v.clone()) for k, v in target_sd.items())
    copy_metadata(target_sd, merged)
    report = dict(base_loaded=[], base_missing_or_mismatch=[], resnet_loaded=[],
                  resnet_missing_or_mismatch=[], random=[], skipped_frozen=[])

    for k, dst in target_sd.items():
        if k in base_sd and shape(base_sd[k]) == shape(dst):
            merged[k] = base_sd[k]
            report['base_loaded'].append(k)
        elif k in base_sd:
            report['base_missing_or_mismatch'].append([k, shape(base_sd[k]), shape(dst)])

    prefix = 'mm_frontend.mf_img_backbone.backbone.'
    for rk, rv in resnet_sd.items():
        tk = prefix + rk
        if tk in target_sd and shape(rv) == shape(target_sd[tk]):
            merged[tk] = rv
            report['resnet_loaded'].append([rk, tk])
        elif tk in target_sd:
            report['resnet_missing_or_mismatch'].append([rk, shape(rv), shape(target_sd[tk])])

    pretrained = set(report['base_loaded']) | {tk for _, tk in report['resnet_loaded']}
    report['random'] = [k for k in target_sd if k not in pretrained]

    ckpt = dict(state_dict=merged)
    meta = base_obj.get('meta', {}) if isinstance(base_obj, dict) else {}
    meta = dict(meta)
    meta['resnet50_merge'] = dict(config=os.path.abspath(args.config),
                                  base=os.path.abspath(args.base),
                                  resnet=os.path.abspath(args.resnet))
    meta['summary'] = dict(base_loaded=len(report['base_loaded']),
                           resnet_loaded=len(report['resnet_loaded']),
                           random=len(report['random']))
    ckpt['meta'] = meta

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(ckpt, args.out)
    with open(args.out + '.report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print('Saved:', args.out)
    print('base_loaded:', len(report['base_loaded']))
    print('resnet_loaded:', len(report['resnet_loaded']))
    print('random:', len(report['random']))
    print('random sample:', report['random'][:30])


if __name__ == '__main__':
    main()
