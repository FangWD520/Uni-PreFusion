# Third-party source notices

This repository integrates and modifies components from several research codebases. The original file headers, copyright statements, and existing license files are retained. The texts collected under `third_party/licenses/` are copies of upstream licenses for the relevant upstream material; they do not relicense this entire repository or establish a new license grant for every local modification.

No repository-wide license grant is added by this notice. In particular, an Apache-2.0 or MIT license belonging to one dependency must not be treated as a license for the original or modified work of a different contributor.

## Framework and fusion sources

| Component / local scope | Upstream source | License evidence retained |
| --- | --- | --- |
| UniLION foundation, including unified sparse temporal and multi-task framework in `projects/` | [happinesslz/UniLION](https://github.com/happinesslz/UniLION) | The inspected upstream root at commit `b75c28415ed53d5474bcb9be776d032c66ec432b` has no root `LICENSE`, `LICENSE.txt`, `LICENSE.md`, or `COPYING` file. No license is inferred for UniLION-specific material. Existing per-file and nested dependency notices remain applicable. |
| MambaFusion-derived frontend and components in `pcdet/`, including `models/mm_backbone/` | [AutoLab-SAI-SJTU/MambaFusion](https://github.com/AutoLab-SAI-SJTU/MambaFusion) | Upstream Apache-2.0 text: [MambaFusion license](third_party/licenses/AutoLab-SAI-SJTU_MambaFusion_LICENSE). The upstream installation text also uses the historical `VERITAS12/MambaFusion` URL. Local modifications and integration are not claimed to match upstream HEAD byte-for-byte. |
| OpenPCDet-derived infrastructure in `pcdet/` | [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | Upstream Apache-2.0 text: [OpenPCDet license](third_party/licenses/open-mmlab_OpenPCDet_LICENSE). Nested source-specific notices are preserved. This attribution does not label all files in `pcdet/` as exclusively OpenPCDet code. |
| VMamba image/state-space components in `pcdet/models/vmamba/` | [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) | Upstream MIT text, copyright MzeroMiko: [VMamba license](third_party/licenses/MzeroMiko_VMamba_LICENSE). This does not override the distinct notices of Mamba-derived CUDA kernels. |
| BEVFusion-derived detection-head material identified in `pcdet/models/dense_heads/transfusion_head.py` | [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) | Upstream Apache-2.0 text: [BEVFusion license](third_party/licenses/mit-han-lab_bevfusion_LICENSE). The adaptation comment is retained. |
| SparseDrive, acknowledged by the UniLION foundation | [swc-17/SparseDrive](https://github.com/swc-17/SparseDrive) | Upstream MIT text: [SparseDrive license](third_party/licenses/swc-17_SparseDrive_LICENSE). This acknowledgement alone does not establish that all local motion/planning modules are verbatim SparseDrive files. |

## Mamba and selective-scan implementations

The separate Mamba variants are kept in their existing directories. Their shared upstream origin does not make their Python packages or compiled operators interchangeable.

- `mamba_diffv/mamba/LICENSE`, `pcdet/ops/mamba/LICENSE`, and `projects/mmdet3d_plugin/models/ops/mamba/LICENSE` retain their existing Apache-2.0 texts and the Tri Dao / Albert Gu attribution. The upstream project is [state-spaces/mamba](https://github.com/state-spaces/mamba). A reference copy of its current upstream Apache-2.0 license is also kept in [third_party/licenses/state-spaces_mamba_LICENSE](third_party/licenses/state-spaces_mamba_LICENSE).
- `selective_scan/setup.py` retains its modification attribution and the Albert Gu / Tri Dao copyright header. Many CUDA source files under `selective_scan/csrc/selective_scan/` retain Tri Dao copyright notices. VMamba-related integration does not remove these underlying notices.
- `selective_scan/csrc/selective_scan/uninitialized_copy.cuh` contains a complete NVIDIA copyright, redistribution conditions, and disclaimer in the source header. Those terms are retained in that file and must not be replaced by the VMamba MIT label.
- A `License :: OSI Approved :: BSD License` classifier in an inherited setup script is not used here as a replacement for the license text and source-specific notices. The retained license files and headers document the actual available evidence.

## OpenMMLab and other file-specific material

- The bundled MMCV copy retains [mmcv/LICENSE](mmcv/LICENSE) and [mmcv/LICENSES.md](mmcv/LICENSES.md). The latter identifies specially licensed operations, including NVIDIA-licensed `upfirdn2d` and `fused_leaky_relu`; MMCV's general Apache-2.0 text does not replace these exceptions.
- The bundled MMDetection3D copy retains [mmdetection3d/LICENSE](mmdetection3d/LICENSE). OpenMMLab attribution in adapted Swin, neck, augmentation, and other files is retained.
- `pcdet/ops/bev_pool_v2/` and `projects/mmdet3d_plugin/models/bev_backbones/custom_resnet.py` retain Phigent Robotics copyright headers. This notice does not assign those copyrights to the Uni-PreFusion authors or infer a new license grant from the header alone.
- The KITTI evaluation utilities retain their separate [MIT license](pcdet/datasets/kitti/kitti_object_eval_python/LICENSE) and the per-file source references, including RRPN-related rotated-IoU attribution.
- Other file-local references and copyright notices, including CenterPoint, PyTorch, NVIDIA, and related research implementations, remain in their source files. This document does not replace those notices or claim an exhaustive line-by-line provenance reconstruction.

## DVR / occupancy rendering

`lib/dvr/dvr.cpp` and `lib/dvr/dvr.cu` identify [tarashakhurana/4d-occ-forecasting](https://github.com/tarashakhurana/4d-occ-forecasting) and retain the line `Modified by Haisong Liu`. The upstream MIT text is preserved in [third_party/licenses/tarashakhurana_4d-occ-forecasting_LICENSE](third_party/licenses/tarashakhurana_4d-occ-forecasting_LICENSE). That upstream text records the original upstream terms; it is not a new statement of permission for independently authored modifications.

## License-source record

The restored license texts were obtained from the official upstream GitHub repositories and checked on 2026-10-21. The table below records the exact upstream commits used for the fetched texts. These commits bind the copied license texts, not the entire locally modified implementation.

| Upstream repository | Inspected commit |
| --- | --- |
| [AutoLab-SAI-SJTU/MambaFusion](https://github.com/AutoLab-SAI-SJTU/MambaFusion) | `d85c2feeb19207b20e42d50421afb11c095f3efa` |
| [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) | `2ed52ead062a51a64521ed3871d52914bf532876` |
| [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | `233f849829b6ac19afb8af8837a0246890908755` |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) | `e9594ce1c732d97440f0332fdc43170a2294dbfa` |
| [tarashakhurana/4d-occ-forecasting](https://github.com/tarashakhurana/4d-occ-forecasting) | `e1ea9434113d303986684395a961aaacac90d470` |
| [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) | `326653dc06e0938edf1aae7d01efcd158ba83de5` |
| [swc-17/SparseDrive](https://github.com/swc-17/SparseDrive) | `fbadea693cbef3f3daea7705e521c0dd3321605c` |

Copyright ownership and any permissions for Uni-PreFusion-specific contributions or released model weights are not determined by this third-party source inventory. No additional grant is implied.
