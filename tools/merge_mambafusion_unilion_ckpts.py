import argparse
import json
import os
import sys
from collections import OrderedDict

import torch
from mmcv import Config
from mmdet.models import build_detector


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _import_plugin(cfg):
    if getattr(cfg, 'plugin', False):
        import importlib
        plugin_dir = cfg.plugin_dir
        module_dir = os.path.dirname(plugin_dir).split('/')
        module_path = module_dir[0]
        for m in module_dir[1:]:
            module_path += '.' + m
        importlib.import_module(module_path)


def _state_dict(ckpt, key_candidates):
    obj = torch.load(ckpt, map_location='cpu')
    for key in key_candidates:
        if isinstance(obj, dict) and key in obj:
            return obj[key], obj
    if isinstance(obj, dict):
        return obj, obj
    raise TypeError(f'Unsupported checkpoint format: {ckpt}')


def _copy_metadata(src, dst, prefix=''):
    metadata = getattr(src, '_metadata', None)
    if metadata is None:
        return
    if not hasattr(dst, '_metadata'):
        dst._metadata = OrderedDict()
    for key, value in metadata.items():
        dst._metadata[prefix + key] = value.copy() if isinstance(value, dict) else value


def _shape(x):
    return tuple(x.shape) if hasattr(x, 'shape') else None


def build_current_model_state(config):
    cfg = Config.fromfile(config)
    _import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    return model.state_dict()


def merge(config, unilion_ckpt, mambafusion_ckpt, out_path):
    model_sd = build_current_model_state(config)
    uni_sd, uni_ckpt = _state_dict(unilion_ckpt, ['state_dict', 'model_state', 'model'])
    mf_sd, mf_ckpt = _state_dict(mambafusion_ckpt, ['model_state', 'state_dict', 'model'])

    merged = OrderedDict((k, v.clone()) for k, v in model_sd.items())
    # Keep metadata from the freshly-built current model state dict. This is
    # important because spconv checkpoint loaders use per-module metadata to
    # decide whether 5D conv weights need legacy-layout conversion. Since merged
    # tensors are written in the current model layout, source checkpoint metadata
    # must not override these entries.
    _copy_metadata(model_sd, merged)
    report = dict(
        model_params=len(model_sd),
        unilion_params=len(uni_sd),
        mambafusion_params=len(mf_sd),
        policy=dict(
            unilion_allowed_prefixes=[
                'pts_voxel_encoder.', 'img_map2bev.', 'pts_backbone.',
                'map2bev.', 'bev_backbone.', 'pts_neck.', 'temporal_fuser.'
            ],
            mambafusion_allowed_targets=[
                'mm_frontend.mf_img_backbone.*', 'mm_frontend.mf_img_neck.*'
            ],
        ),
        unilion_loaded=[],
        unilion_shape_mismatch=[],
        unilion_missing_in_ckpt=[],
        unilion_skipped_by_policy=[],
        mambafusion_loaded=[],
        mambafusion_shape_mismatch=[],
        mambafusion_skipped_by_policy=[],
        still_random_init=[],
    )

    # 1) UniLION checkpoint: only load the requested sparse 3D backbone side.
    allowed_uni_prefixes = (
        'pts_voxel_encoder.', 'img_map2bev.', 'pts_backbone.',
        'map2bev.', 'bev_backbone.', 'pts_neck.', 'temporal_fuser.'
    )
    for k, dst in model_sd.items():
        if not k.startswith(allowed_uni_prefixes):
            report['unilion_skipped_by_policy'].append(k)
            continue
        if k in uni_sd and _shape(uni_sd[k]) == _shape(dst):
            merged[k] = uni_sd[k]
            report['unilion_loaded'].append(k)
        elif k in uni_sd:
            report['unilion_shape_mismatch'].append([k, _shape(uni_sd[k]), _shape(dst)])
        else:
            report['unilion_missing_in_ckpt'].append(k)

    # 2) MambaFusion checkpoint: only load the generic image backbone/neck.
    excluded_mf_prefixes = (
        'vfe.', 'backbone_3d.', 'map_to_bev_module.', 'vtransform.',
        'fuser.', 'backbone_2d.', 'dense_head.')
    report['mambafusion_excluded'] = [k for k in mf_sd if k.startswith(excluded_mf_prefixes)]

    def try_load(src_k, src_v, candidates):
        dst_k = next((k for k in candidates if k in model_sd), candidates[0])
        if dst_k not in model_sd:
            report['mambafusion_shape_mismatch'].append([src_k, dst_k, _shape(src_v), None])
            return
        src_shape, dst_shape = _shape(src_v), _shape(model_sd[dst_k])
        if src_shape == dst_shape:
            merged[dst_k] = src_v
            report['mambafusion_loaded'].append([src_k, dst_k])
        elif len(src_shape or []) == 5 and tuple(src_v.permute(0, 2, 3, 4, 1).shape) == dst_shape:
            merged[dst_k] = src_v.permute(0, 2, 3, 4, 1).contiguous()
            report['mambafusion_loaded'].append([src_k, dst_k, 'permute_0_2_3_4_1'])
        elif len(src_shape or []) == 5 and tuple(src_v.permute(4, 0, 1, 2, 3).shape) == dst_shape:
            merged[dst_k] = src_v.permute(4, 0, 1, 2, 3).contiguous()
            report['mambafusion_loaded'].append([src_k, dst_k, 'permute_4_0_1_2_3'])
        else:
            report['mambafusion_shape_mismatch'].append([src_k, dst_k, src_shape, dst_shape])

    for src_k, src_v in mf_sd.items():
        candidates = []
        if src_k.startswith('mm_backbone.'):
            tail = src_k[len('mm_backbone.'):]
            if tail.startswith('backbone_vssm.'):
                candidates.append('mm_frontend.mf_img_backbone.backbone_vssm.' + tail[len('backbone_vssm.'):])
            elif tail.startswith('vssm_down_block.'):
                candidates.append('mm_frontend.mf_img_backbone.down_block.' + tail[len('vssm_down_block.'):])
            else:
                report['mambafusion_skipped_by_policy'].append(src_k)
        elif src_k.startswith('neck.'):
            tail = src_k[len('neck.'):]
            candidates.append('mm_frontend.mf_img_neck.neck.' + tail)
        else:
            report['mambafusion_skipped_by_policy'].append(src_k)
        if candidates:
            try_load(src_k, src_v, candidates)


    pretrained = set(report['unilion_loaded']) | {
        entry[1] for entry in report['mambafusion_loaded'] if len(entry) >= 2
    }
    report['still_random_init'] = [k for k in model_sd if k not in pretrained]

    ckpt = dict(
        state_dict=merged,
        meta=dict(
            merged_from=dict(
                unilion=os.path.abspath(unilion_ckpt),
                mambafusion=os.path.abspath(mambafusion_ckpt),
                config=os.path.abspath(config),
            ),
            merge_policy=dict(
                unilion_allowed_prefixes=list(allowed_uni_prefixes),
                mambafusion_allowed_targets=[
                    'mm_frontend.mf_img_backbone.*',
                    'mm_frontend.mf_img_neck.*',
                ],
                excluded_task_heads=True,
            ),
            merge_report_summary=dict(
                unilion_loaded=len(report['unilion_loaded']),
                mambafusion_loaded=len(report['mambafusion_loaded']),
                still_random_init=len(report['still_random_init']),
            ),
        ),
    )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, out_path)
    with open(out_path + '.report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('Saved:', out_path)
    print('Report:', out_path + '.report.json')
    print('UniLION loaded:', len(report['unilion_loaded']))
    print('MambaFusion loaded:', len(report['mambafusion_loaded']))
    print('UniLION skipped by policy:', len(report['unilion_skipped_by_policy']))
    print('MambaFusion skipped by policy:', len(report['mambafusion_skipped_by_policy']))
    print('Still random/init-only:', len(report['still_random_init']))
    print('Random/init-only sample:', report['still_random_init'][:30])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/unilion_swin_384_det.py')
    parser.add_argument('--unilion', default='ckpts/unilion_swin_384_seq_e2e.pth')
    parser.add_argument('--mambafusion', default='ckpts/checkpoint_epoch_10.pth')
    parser.add_argument('--out', default='ckpts/unilion_swin_384_det_mambafusion_init.pth')
    args = parser.parse_args()
    merge(args.config, args.unilion, args.mambafusion, args.out)


if __name__ == '__main__':
    main()

