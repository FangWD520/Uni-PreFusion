# Installation

Use Linux, an NVIDIA GPU, a compatible driver, the CUDA **12.4 toolkit including `nvcc`**, and a C++ compiler. The intended stack is Python 3.9, PyTorch 2.5.0+cu124 and torchvision 0.20.0+cu124. A100/A800 are the initial reproduction targets; support for other GPU architectures is not established here.

This repository bundles modified MMCV 1.7.0 and MMDetection3D 1.0.0rc6 sources. Do not substitute current upstream packages. This public packaging has been checked statically; this exact installation sequence has **not** been newly validated by a clean build and complete five-task evaluation. See [reproducibility](reproducibility.md).

## Environment

Run commands from the repository root unless a command changes directory.

```bash
conda create -n uniprefusion python=3.9 -y
conda activate uniprefusion
python -m pip install 'pip<25' 'setuptools<70' wheel ninja packaging
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-runtime.txt
python -m pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
python -m pip install 'causal-conv1d>=1.4.0,<1.5' --no-build-isolation
```

`spconv-cu124==2.3.8` is selected for CUDA 12.4; let its package metadata resolve matching `cumm-cu124`. Do not mix multiple CUDA variants of spconv/cumm. CUDA 12.4 wheels are documented in the [spconv changelog](https://github.com/traveller59/spconv/blob/master/CHANGELOG.md) and [PyPI package](https://pypi.org/project/spconv-cu124/). The inherited `requirement.txt` is a historical environment snapshot containing CUDA 11.6 packages and unrelated tools, not the installation entry point.

If a matching binary wheel is unavailable, build against the same Python/PyTorch/CUDA ABI. A wheel compiled for another ABI is not interchangeable. Record the resolved versions with any experiment.

## Local operators

Adjust `MAX_JOBS` to available host memory. `TORCH_CUDA_ARCH_LIST=8.0` targets A100/A800; use the architecture of your GPU and rebuild dependencies when changing the stack.

```bash
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

(cd mmcv && MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install -v -e . --no-deps --no-build-isolation)
(cd mmdetection3d && python -m pip install -v -e . --no-deps --no-build-isolation)
(cd projects && FORCE_CUDA=1 python setup.py develop)
python setup_mf_ops.py build_ext --inplace
```

`projects/setup.py` builds BEV pooling and RWKV operators. `setup_mf_ops.py` builds ingroup-index, flattened-window and PointNet2-stack operators for the fusion frontend. The occupancy evaluator also JIT-compiles `lib/dvr/dvr.cpp` and `lib/dvr/dvr.cu` when the dataset/evaluator is imported. Keep those files and run from the repository root; a first import may take time.

## Separate Mamba implementations

| Source | Python package / extension | Role |
|---|---|---|
| `mamba_diffv/mamba` | `mamba_ssm` / `selective_scan_cuda` | MambaFusion implementation, including `Block2` |
| `projects/mmdet3d_plugin/models/ops/mamba` | `unilion_mamba_ssm` / `unilion_selective_scan_cuda` | Isolated original UniLION bidirectional implementation |
| `selective_scan` | `selective_scan_cuda_core`, `selective_scan_cuda_oflex`, `selective_scan_cuda_ndstate` | Additional VMamba kernels |

```bash
(cd mamba_diffv/mamba && MAMBA_FORCE_BUILD=TRUE python -m pip install -v -e . --no-deps --no-build-isolation)
(cd projects/mmdet3d_plugin/models/ops/mamba && MAMBA_FORCE_BUILD=TRUE python -m pip install -v -e . --no-deps --no-build-isolation)
(cd selective_scan && python -m pip install -v . --no-deps --no-build-isolation)

export UNILION_MAMBA_BACKEND=external
export PYTHONPATH="$PWD:$PWD/projects:$PWD/mamba_diffv/mamba:$PWD/projects/mmdet3d_plugin/models/ops/mamba:$PWD/selective_scan${PYTHONPATH:+:$PYTHONPATH}"
```

Keep `UNILION_MAMBA_BACKEND=external` for the released checkpoint. This preserves the supplied trained structure. Switching to `auto`/`unilion` changes the selected block and can introduce backward-branch parameters. Do not install the duplicate `pcdet/ops/mamba` package over `mamba_diffv/mamba` or replace these sources with the latest generic `mamba-ssm` release.

## Checks

Download the release assets and follow [data preparation](data_preparation.md).

```bash
python tools/check_release.py                  # standard library only
python tools/check_release.py --check-assets   # require checkpoint + four tables
python tools/check_release.py --hash-assets    # full SHA-256; reads several GB
python tools/check_release.py --check-assets --check-imports
```

The default command checks paths, syntax, config inheritance and small anchor hashes without importing PyTorch or CUDA. `--check-imports` explicitly loads installed packages/native extensions. A successful check is not a model forward pass, numerical consistency result or reproduction of paper metrics.
