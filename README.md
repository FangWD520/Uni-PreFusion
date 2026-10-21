# Uni-PreFusion

## A Height-Aware Pre-Lifting LiDAR-Camera Fusion Frontend for Unified Autonomous Driving

**Xuerui Fang, Gulimila Kezierbieke, Chunyang Chen, Congcong Bian, Zehao Xue, Yeerjiang Halimu, Hui Li**

Xinjiang Agricultural University · Jiangnan University

[中文介绍](README_zh-CN.md) · [Installation](docs/INSTALL.md) · [Data preparation](docs/data_preparation.md) · [Checkpoints](docs/checkpoints.md) · [Training](docs/training.md)

Uni-PreFusion lets LiDAR geometry guide camera features **before depth-based lifting into 3D**. A height-aware sparse-to-pillar interface connects dense image interaction to a sparse temporal driving backbone, supporting **3D detection, BEV map segmentation, occupancy prediction, motion forecasting, and ego planning**.

![Uni-PreFusion architecture](figures/overview.png)

### Method

Post-lifting fusion must reason with camera features whose 3D positions already depend on estimated depth. Uni-PreFusion introduces geometric interaction earlier in the pipeline:

1. **Height-aware pillar tokens.** Sparse LiDAR voxels are grouped into pillars. Learned height embeddings and within-pillar weighted aggregation retain vertical cues during compression.
2. **Pre-lifting interaction.** Pillar tokens interact with dense multi-view image features through a MambaFusion-style fusion backbone. LiDAR geometry refines image features before view transformation.
3. **Residual voxel restoration.** Refined pillar context is broadcast back to the original sparse voxels through a residual connection, retaining their coordinates and voxel-level representation.
4. **Unified temporal reasoning.** Refined image features are lifted into 3D and combined with sparse LiDAR features. A UniLION-based temporal backbone supplies a shared representation to the five task heads.

The height-aware pre-lifting interface is the focus of this work. The UniLION backbone, MambaFusion interaction blocks, and VMamba image encoder are inherited components; see [third-party notices](THIRD_PARTY_NOTICES.md).

<details>
<summary>Height-aware interface details</summary>

![Height-aware pillar adaptation and residual restoration](figures/detail.png)

</details>

### Manuscript-reported results

nuScenes validation, image resolution **704 × 256**. **L**, **C**, and **T** denote LiDAR, camera, and temporal inputs. The tables reproduce the manuscript's main comparison; they are not new measurements from packaging this repository.

| Setting | NDS ↑ | Detection mAP ↑ | BEV map mIoU ↑ | Occupancy RayIoU ↑ |
|:--|--:|--:|--:|--:|
| LC | 74.2 | 71.9 | 70.4 | 50.2 |
| LCT | 74.8 | 72.4 | 72.4 | 51.7 |

| Setting | Car minADE ↓ (m) | Pedestrian minADE ↓ (m) | Planning L2 ↓ (m) | Collision ↓ (%) |
|:--|--:|--:|--:|--:|
| LC | 0.61 | 0.38 | 0.65 | 0.23 |
| LCT | 0.57 | 0.36 | 0.62 | 0.23 |

Planning uses the manuscript's open-loop protocol without ego-status inputs or navigation commands. The model has approximately **81.7M parameters**. This code release does not assert paper acceptance, a DOI, or a paper identifier.

### Get started

The reference stack is **Linux, Python 3.9, PyTorch 2.5.0 with CUDA 12.4, and torchvision 0.20.0**. Build CUDA extensions for the installed PyTorch/CUDA combination by following [Installation](docs/INSTALL.md).

```bash
git clone https://github.com/FangWD520/Uni-PreFusion.git
cd Uni-PreFusion

# Download the full LCT checkpoint and all four required Hilbert tables.
python tools/download_models.py

# Check the packaged files without importing CUDA extensions.
python tools/check_release.py
```

The checkpoint and tables are distributed through [GitHub Releases](https://github.com/FangWD520/Uni-PreFusion/releases/tag/v1.0.0), with byte counts and SHA-256 checksums in [the model manifest](ckpts/model_manifest.json). They are not stored in Git history. Two small motion/planning constructor anchors are included under `assets/`.

After preparing the task annotations described in [Data preparation](docs/data_preparation.md):

```bash
export UNILION_MAMBA_BACKEND=external
export PYTHONPATH="$PWD:$PWD/projects:$PWD/mamba_diffv/mamba:$PYTHONPATH"

# Argument order: CONFIG CHECKPOINT NUMBER_OF_GPUS.
bash tools/dist_test.sh configs/uniprefusion_seq_e2e.py \
    ckpts/uniprefusion-final.pth 1 --eval mAP
```

The final configuration keeps all five tasks enabled. A single-GPU invocation is shown for syntax; memory requirements depend on task outputs, scene history, and installed kernels. [Training](docs/training.md) explains the staged entry points and initialization artifacts required for training.

### Code and checkpoint compatibility

- Internal `UniLION` registration names, Python paths, and state-dictionary keys are retained for checkpoint compatibility. Public entry points use `configs/uniprefusion_*.py` names.
- This checkpoint uses the historical **`external` Mamba backend**. Switching to bidirectional `auto` changes the architecture and is not a drop-in replacement. The two Mamba implementations have separate package/extension names; see [Installation](docs/INSTALL.md).
- This is the authors' original code/checkpoint snapshot. Later sparsification experiments and diagnostic runtime repairs are not included. Exact cross-process numerical reproducibility has not been established for the original frontend; see [Reproducibility notes](docs/reproducibility.md).
- Packaging checks do not establish fresh reproduction of every manuscript metric.

### Citation

If you use this work, please cite the manuscript and the underlying methods as appropriate:

```bibtex
@misc{fang_uniprefusion,
  title = {Uni-PreFusion: A Height-Aware Pre-Lifting LiDAR-Camera Fusion Frontend for Unified Autonomous Driving},
  author = {Fang, Xuerui and Kezierbieke, Gulimila and Chen, Chunyang and Bian, Congcong and Xue, Zehao and Halimu, Yeerjiang and Li, Hui},
  howpublished = {\url{https://github.com/FangWD520/Uni-PreFusion}},
  note = {Manuscript and accompanying code}
}
```

### Acknowledgements

We thank [UniLION](https://github.com/happinesslz/UniLION), [LION](https://github.com/happinesslz/LION), MambaFusion, VMamba, [Mamba](https://github.com/state-spaces/mamba), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), and the related driving-task libraries. Existing copyright and license notices remain applicable. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); this repository does not relicense third-party components under one blanket license.
