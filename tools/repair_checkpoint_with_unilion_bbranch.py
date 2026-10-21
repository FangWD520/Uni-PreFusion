import argparse
import json
import os
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from mmcv import Config
from mmdet.models import build_detector


def _state_dict(path):
    obj = torch.load(path, map_location='cpu')
    for key in ['state_dict', 'model_state', 'model']:
        if isinstance(obj, dict) and key in obj:
            return obj[key], obj
    return obj, obj


def _shape(x):
    return tuple(x.shape) if hasattr(x, 'shape') else None


def _import_plugin(cfg):
    if getattr(cfg, 'plugin', False):
        import importlib
        plugin_dir = cfg.plugin_dir
        module_dir = os.path.dirname(plugin_dir).split('/')
        module_path = module_dir[0]
        for m in module_dir[1:]:
            module_path += '.' + m
        importlib.import_module(module_path)


def _copy_metadata(src, dst):
    md = getattr(src, '_metadata', None)
    if md is not None:
        dst._metadata = md.copy()


def build_model_state(config):
    cfg = Config.fromfile(config)
    _import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    # Build the exact target structure from the formal config. UniLION Mamba is
    # isolated as unilion_mamba_ssm, while MambaFusion keeps global mamba_ssm,
    # so we no longer need to disable the original MambaFusion mm_backbone here.
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    return model.state_dict()


def is_bbranch_key(key):
    patterns = ('.A_b_log', '.D_b', '.conv1d_b.', '.x_proj_b.', '.dt_proj_b.')
    return key.startswith('pts_backbone.') and any(p in key for p in patterns)


def main():
    parser = argparse.ArgumentParser(
        description='Keep a trained checkpoint, and fill only missing UniLION bidirectional Mamba _b branch weights.')
    parser.add_argument('--config', default='projects/configs/unilion_swin_384_seq_e2e.py')
    parser.add_argument('--trained', required=True, help='your previously trained checkpoint, e.g. ckpts/final.pth')
    parser.add_argument('--unilion', default='ckpts/unilion_swin_384_seq_e2e.pth', help='official UniLION checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--fill-all-missing-unilion', action='store_true',
                        help='Dangerous: fill every missing key from official UniLION, not only _b branch.')
    args = parser.parse_args()

    model_sd = build_model_state(args.config)
    trained_sd, _ = _state_dict(args.trained)
    uni_sd, _ = _state_dict(args.unilion)

    merged = OrderedDict((k, v.clone()) for k, v in model_sd.items())
    _copy_metadata(model_sd, merged)
    report = dict(
        trained_loaded=[],
        trained_mismatch=[],
        unilion_filled=[],
        unilion_mismatch=[],
        still_random=[],
        preserved_existing_heads=[],
    )

    # 1) Preserve everything possible from your trained checkpoint first.
    for k, dst in model_sd.items():
        if k in trained_sd and _shape(trained_sd[k]) == _shape(dst):
            merged[k] = trained_sd[k]
            report['trained_loaded'].append(k)
        elif k in trained_sd:
            report['trained_mismatch'].append([k, _shape(trained_sd[k]), _shape(dst)])

    trained_loaded = set(report['trained_loaded'])

    # 2) Fill only keys absent in the trained checkpoint, defaulting to the
    # official UniLION bidirectional Mamba backward branch. Do not overwrite
    # already-trained det/map/occ/motion/planning heads.
    for k, dst in model_sd.items():
        if k in trained_loaded:
            if k.startswith(('det_head.', 'map_head.', 'occ_head.', 'motion_head.', 'planning_head.')):
                report['preserved_existing_heads'].append(k)
            continue
        if not (args.fill_all_missing_unilion or is_bbranch_key(k)):
            continue
        if k in uni_sd and _shape(uni_sd[k]) == _shape(dst):
            merged[k] = uni_sd[k]
            report['unilion_filled'].append(k)
        elif k in uni_sd:
            report['unilion_mismatch'].append([k, _shape(uni_sd[k]), _shape(dst)])

    pretrained = set(report['trained_loaded']) | set(report['unilion_filled'])
    report['still_random'] = [k for k in model_sd if k not in pretrained]

    ckpt = dict(
        state_dict=merged,
        meta=dict(
            merged_from=dict(
                config=os.path.abspath(args.config),
                trained=os.path.abspath(args.trained),
                unilion=os.path.abspath(args.unilion),
            ),
            summary=dict(
                trained_loaded=len(report['trained_loaded']),
                unilion_filled=len(report['unilion_filled']),
                still_random=len(report['still_random']),
                preserved_existing_heads=len(report['preserved_existing_heads']),
            ),
        ),
    )
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, args.out)
    with open(args.out + '.report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('Saved:', args.out)
    print('trained_loaded:', len(report['trained_loaded']))
    print('unilion_filled:', len(report['unilion_filled']))
    print('still_random:', len(report['still_random']))
    print('preserved_existing_heads:', len(report['preserved_existing_heads']))
    print('unilion_filled sample:', report['unilion_filled'][:30])
    print('still_random sample:', report['still_random'][:30])


if __name__ == '__main__':
    main()
