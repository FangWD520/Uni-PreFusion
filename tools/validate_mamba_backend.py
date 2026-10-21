import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception as exc:
        print('torch import failed; checkpoint comparison skipped:', repr(exc))
        return False


def _state_dict_keys(path):
    import torch
    obj = torch.load(path, map_location='cpu')
    for key in ('state_dict', 'model_state', 'model'):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
            break
    return set(obj.keys())


def _set_backend(backend):
    if backend is None:
        os.environ.pop('UNILION_MAMBA_BACKEND', None)
    else:
        os.environ['UNILION_MAMBA_BACKEND'] = backend
    for name in list(sys.modules):
        if name == 'projects.mmdet3d_plugin.models.utils.mamba_compat':
            sys.modules.pop(name)


def inspect_backend(backend):
    _set_backend(backend)
    mod = importlib.import_module('projects.mmdet3d_plugin.models.utils.mamba_compat')
    block = mod.MambaBlock(d_model=8, d_state=2, d_conv=2, expand=1)
    keys = set(block.state_dict().keys())
    b_keys = sorted(k for k in keys if any(p in k for p in (
        'A_b_log', 'D_b', 'conv1d_b.', 'x_proj_b.', 'dt_proj_b.')))
    info = dict(mod.MAMBA_BACKEND_INFO)
    info.update(
        env=backend or '<unset>',
        block_class=mod.MambaBlock.__name__,
        block_keys=len(keys),
        b_branch_keys=b_keys,
        has_b_branch=bool(b_keys),
    )
    return info


def print_info(label, info):
    print('\n[{}]'.format(label))
    for key in ('env', 'requested', 'selected', 'bidirectional', 'block_module',
                'block_file', 'block_class', 'block_keys', 'has_b_branch'):
        print('{}: {}'.format(key, info.get(key)))
    if info.get('b_branch_keys'):
        print('b_branch_key_sample:', info['b_branch_keys'][:8])


def compare_checkpoint(path):
    if not path or not os.path.exists(path) or not _has_torch():
        return
    ckpt_keys = _state_dict_keys(path)
    b_patterns = ('.A_b_log', '.D_b', '.conv1d_b.', '.x_proj_b.', '.dt_proj_b.')
    ckpt_b = sorted(k for k in ckpt_keys if k.startswith('pts_backbone.') and any(p in k for p in b_patterns))
    ckpt_fwd = sorted(k for k in ckpt_keys if k.startswith('pts_backbone.') and '.mamba.' in k and not any(p in k for p in b_patterns))
    print('\n[checkpoint {}]'.format(path))
    print('pts_backbone forward/simple mamba keys:', len(ckpt_fwd))
    print('pts_backbone backward-branch keys:', len(ckpt_b))
    print('has previous one-way structure:', len(ckpt_fwd) > 0 and len(ckpt_b) == 0)
    if ckpt_b:
        print('b_branch_checkpoint_sample:', ckpt_b[:8])


def main():
    parser = argparse.ArgumentParser(description='Inspect UniLION Mamba backend selection.')
    parser.add_argument('--checkpoint', default='ckpts/final.pth')
    args = parser.parse_args()

    default_info = inspect_backend(None)
    unilion_info = inspect_backend('auto')
    print_info('default backend', default_info)
    print_info('explicit auto/unilion backend', unilion_info)

    assert default_info['selected'] == 'external', default_info
    assert not default_info['has_b_branch'], default_info
    assert unilion_info['selected'] == 'unilion', unilion_info
    assert unilion_info['has_b_branch'], unilion_info
    compare_checkpoint(args.checkpoint)
    print('\nMamba backend switch validation passed.')


if __name__ == '__main__':
    main()
