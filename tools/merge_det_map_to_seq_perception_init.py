import argparse
import json
import os
from collections import OrderedDict

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
        importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))


def _copy_metadata(src, dst):
    md = getattr(src, '_metadata', None)
    if md is not None:
        dst._metadata = md.copy()


def build_model_state(config):
    cfg = Config.fromfile(config)
    _import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    return model.state_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/unilion_swin_384_seq_perception.py')
    parser.add_argument('--det-map', required=True, help='previous-stage det+map checkpoint')
    parser.add_argument('--unilion', default='ckpts/unilion_swin_384_seq_e2e.pth', help='source for temporal/occ weights')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    model_sd = build_model_state(args.config)
    det_map_sd, _ = _state_dict(args.det_map)
    uni_sd, _ = _state_dict(args.unilion)
    merged = OrderedDict((k, v.clone()) for k, v in model_sd.items())
    _copy_metadata(model_sd, merged)
    report = dict(det_map_loaded=[], unilion_seq_loaded=[], mismatch=[], missing_or_random=[])

    # 1) Load everything possible from previous det+map stage.
    for k, dst in model_sd.items():
        if k in det_map_sd and _shape(det_map_sd[k]) == _shape(dst):
            merged[k] = det_map_sd[k]
            report['det_map_loaded'].append(k)
        elif k in det_map_sd:
            report['mismatch'].append(['det_map', k, _shape(det_map_sd[k]), _shape(dst)])

    # 2) Fill seq/occ-specific modules from official UniLION seq/e2e checkpoint.
    seq_prefixes = ('temporal_fuser.', 'occ_head.')
    for k, dst in model_sd.items():
        if not k.startswith(seq_prefixes):
            continue
        if k in uni_sd and _shape(uni_sd[k]) == _shape(dst):
            merged[k] = uni_sd[k]
            report['unilion_seq_loaded'].append(k)
        elif k in uni_sd:
            report['mismatch'].append(['unilion_seq', k, _shape(uni_sd[k]), _shape(dst)])

    pretrained = set(report['det_map_loaded']) | set(report['unilion_seq_loaded'])
    report['missing_or_random'] = [k for k in model_sd if k not in pretrained]

    ckpt = dict(
        state_dict=merged,
        meta=dict(
            merged_from=dict(
                config=os.path.abspath(args.config),
                det_map=os.path.abspath(args.det_map),
                unilion=os.path.abspath(args.unilion),
            ),
            summary=dict(
                det_map_loaded=len(report['det_map_loaded']),
                unilion_seq_loaded=len(report['unilion_seq_loaded']),
                missing_or_random=len(report['missing_or_random']),
            ),
        ),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    with open(args.out + '.report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Saved:', args.out)
    print('det_map_loaded:', len(report['det_map_loaded']))
    print('unilion_seq_loaded:', len(report['unilion_seq_loaded']))
    print('missing_or_random:', len(report['missing_or_random']))
    print('random sample:', report['missing_or_random'][:30])


if __name__ == '__main__':
    main()
