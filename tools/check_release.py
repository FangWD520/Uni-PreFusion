#!/usr/bin/env python3
"""Validate release structure/assets without loading a model or checkpoint.

Default execution uses only the Python standard library. Native imports are
performed exclusively with --check-imports; no torch.load/pickle.load is used.
"""
import argparse
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys


STAGES = {
    'det': dict(det=True, map=False, occ=False, motion=False, planning=False),
    'det_map': dict(det=True, map=True, occ=False, motion=False, planning=False),
    'seq_perception': dict(det=True, map=True, occ=True, motion=False, planning=False),
    'seq_e2e': dict(det=True, map=True, occ=True, motion=True, planning=True),
}
ANCHORS = {
    'assets/kmeans_motion_6.npy': (10, 6, 12, 2),
    'assets/kmeans_planning_6.npy': (3, 6, 6, 2),
}
LARGE = {'ckpts/uniprefusion-final.pth'} | {
    'ckpts/hilbert_template/curve_template_3d_rank_{}.pth'.format(r)
    for r in (7, 8, 9, 10)
}
SOURCE_FILES = [
    'setup_mf_ops.py', 'requirements-runtime.txt', 'tools/train.py',
    'tools/test.py', 'tools/dist_train.sh', 'tools/dist_test.sh',
    'tools/data_converter/nuscenes_converter.py',
    'projects/setup.py', 'projects/mmdet3d_plugin/models/lion.py',
    'projects/mmdet3d_plugin/models/utils/mamba_compat.py',
    'projects/mmdet3d_plugin/models/backbones/mambafusion_mm_backbone_frontend.py',
    'projects/mmdet3d_plugin/models/backbones/mambafusion_vmamba_img_backbone.py',
    'projects/mmdet3d_plugin/models/ops/mamba/setup.py',
    'mamba_diffv/mamba/setup.py', 'selective_scan/setup.py',
    'mmcv/setup.py', 'mmdetection3d/setup.py',
    'lib/dvr/dvr.cpp', 'lib/dvr/dvr.cu', 'ckpts/model_manifest.json',
]


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('asset path must be relative')
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError('path escapes repository')
    return target


def value(node):
    """Read only literal config declarations; never execute a config."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == 'dict' and not node.args:
            if any(k.arg is None for k in node.keywords):
                raise ValueError('unpacking is not a config literal')
            return {k.arg: value(k.value) for k in node.keywords}
    if isinstance(node, ast.List):
        return [value(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(value(x) for x in node.elts)
    if isinstance(node, ast.Dict):
        return {value(k): value(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def assignment(tree, name):
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in item.targets):
            return value(item.value)
    raise ValueError('missing declaration: ' + name)


def check_anchor(path, expected_shape, item):
    # Parse the NPY header and float32 payload directly, without numpy or pickle.
    with path.open('rb') as handle:
        if handle.read(6) != b'\x93NUMPY':
            raise ValueError('not an NPY file')
        major, minor = handle.read(2)
        if major == 1:
            header_size = struct.unpack('<H', handle.read(2))[0]
        elif major in (2, 3):
            header_size = struct.unpack('<I', handle.read(4))[0]
        else:
            raise ValueError('unsupported NPY header version')
        if header_size > 65536:
            raise ValueError('oversized NPY header')
        metadata = ast.literal_eval(handle.read(header_size).decode(
            'utf-8' if major == 3 else 'latin1'))
        if tuple(metadata['shape']) != expected_shape:
            raise ValueError('anchor shape mismatch')
        if metadata['descr'] != '<f4' or metadata['fortran_order'] is not False:
            raise ValueError('expected little-endian C-order float32 anchor')
        expected_bytes = math.prod(expected_shape) * 4
        data = handle.read(expected_bytes + 1)
        if len(data) != expected_bytes:
            raise ValueError('anchor payload length mismatch')
        if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', data)):
            raise ValueError('nonfinite anchor')
        if hashlib.sha256(data).hexdigest() != item['tensor_sha256']:
            raise ValueError('anchor tensor hash mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check-assets', action='store_true',
                        help='require all large release downloads and check sizes')
    parser.add_argument('--hash-assets', action='store_true',
                        help='also hash every large asset; implies --check-assets')
    parser.add_argument('--check-imports', action='store_true',
                        help='explicitly import packages/extensions; can load CUDA libraries')
    args = parser.parse_args()
    root = args.root.resolve()
    failures, configs, assets, imports = [], [], [], []
    checked_python = set()

    def parse_file(relative):
        p = contained(root, relative)
        tree = ast.parse(p.read_text(encoding='utf-8-sig'), filename=relative)
        checked_python.add(relative)
        return tree

    for relative in SOURCE_FILES:
        try:
            p = contained(root, relative)
            if not p.is_file():
                raise ValueError('required source/resource missing')
            if p.suffix == '.py':
                parse_file(relative)
        except (OSError, ValueError, SyntaxError) as exc:
            failures.append({'path': relative, 'error': type(exc).__name__})

    for stage, expected_tasks in STAGES.items():
        public = 'configs/uniprefusion_{}.py'.format(stage)
        legacy = 'projects/configs/unilion_swin_384_{}.py'.format(stage)
        try:
            tree = parse_file(public)
            base = assignment(tree, '_base_')
            if base != ['../' + legacy]:
                raise ValueError('unexpected config inheritance')
            if (root / public).parent.joinpath(base[0]).resolve() != contained(root, legacy):
                raise ValueError('base path mismatch')
            if assignment(parse_file(legacy), 'tasks') != expected_tasks:
                raise ValueError('stage task set changed')
            if assignment(tree, 'load_from') is not None or assignment(tree, 'resume_from') is not None:
                raise ValueError('implicit checkpoint initialization')
            hooks = assignment(tree, 'log_config')['hooks']
            if hooks != [dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')]:
                raise ValueError('unexpected default logging hooks')
            if assignment(tree, 'work_dir') != 'outputs/Uni-PreFusion/' + stage:
                raise ValueError('unexpected output path')
            if stage == 'seq_e2e' and assignment(tree, 'model') != dict(
                    motion_head=dict(motion_anchor='assets/kmeans_motion_6.npy'),
                    planning_head=dict(planning_anchor='assets/kmeans_planning_6.npy')):
                raise ValueError('final constructor override mismatch')
            configs.append({'path': public, 'tasks': expected_tasks, 'status': 'passed'})
        except (OSError, ValueError, SyntaxError, KeyError, TypeError) as exc:
            failures.append({'path': public, 'error': type(exc).__name__})

    try:
        manifest = json.loads((root / 'ckpts/model_manifest.json').read_text(encoding='utf-8-sig'))
        if manifest['schema'] != 'uniprefusion_release_assets_v1':
            raise ValueError('unexpected manifest schema')
        anchor_items = manifest['constructor_anchors']
        large_items = manifest['assets']
        if len(anchor_items) != 2 or {x['path'] for x in anchor_items} != set(ANCHORS):
            raise ValueError('expected two unique constructor anchors')
        if len(large_items) != 5 or {x['path'] for x in large_items} != LARGE:
            raise ValueError('expected checkpoint and four unique Hilbert assets')
        for item in anchor_items + large_items:
            relative = item['path']
            is_anchor = relative in ANCHORS
            required = is_anchor or args.check_assets or args.hash_assets
            p = contained(root, relative)
            row = {'path': relative, 'required': bool(required)}
            if not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
                raise ValueError('invalid SHA-256 field')
            if not isinstance(item['bytes'], int) or item['bytes'] <= 0:
                raise ValueError('invalid size field')
            if not p.is_file():
                row['status'] = 'missing_required' if required else 'not_downloaded_optional'
                if required:
                    failures.append({'path': relative, 'error': 'MissingAsset'})
            elif p.stat().st_size != item['bytes']:
                row['status'] = 'size_mismatch'
                failures.append({'path': relative, 'error': 'AssetSizeMismatch'})
            elif is_anchor or args.hash_assets:
                if file_sha(p) != item['sha256']:
                    row['status'] = 'sha256_mismatch'
                    failures.append({'path': relative, 'error': 'AssetHashMismatch'})
                else:
                    if is_anchor:
                        check_anchor(p, ANCHORS[relative], item)
                    row['status'] = 'sha256_passed'
            else:
                row['status'] = 'size_passed_sha256_not_requested'
            assets.append(row)
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        failures.append({'path': 'ckpts/model_manifest.json', 'error': type(exc).__name__})

    if args.check_imports:
        # This is the only branch which imports non-standard/native modules.
        roots = ['.', 'projects', 'mamba_diffv/mamba',
                 'projects/mmdet3d_plugin/models/ops/mamba', 'selective_scan']
        for relative in reversed(roots):
            sys.path.insert(0, str(contained(root, relative)))
        if os.environ.get('UNILION_MAMBA_BACKEND', 'external').lower() != 'external':
            failures.append({'setting': 'UNILION_MAMBA_BACKEND', 'error': 'ExpectedExternalBackend'})
        names = ['torch', 'torchvision', 'torch_scatter', 'mmcv', 'mmcv._ext',
                 'mmdet', 'mmdet3d', 'spconv.pytorch', 'causal_conv1d',
                 'mamba_ssm', 'unilion_mamba_ssm', 'selective_scan_cuda',
                 'unilion_selective_scan_cuda', 'selective_scan_cuda_core',
                 'selective_scan_cuda_oflex', 'selective_scan_cuda_ndstate',
                 'pcdet.ops.ingroup_inds.ingroup_inds_cuda',
                 'pcdet.ops.win_coors.flattened_window_cuda',
                 'pcdet.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda']
        expected_versions = {'torch': '2.5.0+cu124', 'torchvision': '0.20.0+cu124',
                             'mmcv': '1.7.0', 'mmdet': '2.28.2',
                             'mmdet3d': '1.0.0rc6'}
        for name in names:
            try:
                module = importlib.import_module(name)
                if name == 'mamba_ssm' and not hasattr(module, 'Block2'):
                    raise ValueError('external MambaFusion Block2 unavailable')
                version = str(getattr(module, '__version__', 'not-declared'))
                if name in expected_versions and version != expected_versions[name]:
                    raise ValueError('package version differs from documented stack')
                if name in ('mamba_ssm', 'unilion_mamba_ssm'):
                    source = 'mamba_diffv/mamba' if name == 'mamba_ssm' else (
                        'projects/mmdet3d_plugin/models/ops/mamba')
                    Path(module.__file__).resolve().relative_to(contained(root, source))
                imports.append({'module': name, 'status': 'imported',
                                'version': version})
            except Exception as exc:
                imports.append({'module': name, 'status': 'failed', 'error': type(exc).__name__})
                failures.append({'module': name, 'error': type(exc).__name__})

    print(json.dumps({'status': 'failed' if failures else 'passed_static_release_checks',
                      'model_forward_executed': False, 'checkpoint_deserialized': False,
                      'native_imports_requested': args.check_imports,
                      'python_files_parsed': len(checked_python), 'configs': configs,
                      'assets': assets, 'imports': imports, 'failures': failures}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
