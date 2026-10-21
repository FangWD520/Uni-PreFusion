# Data and model resources

The final model enables detection, BEV map segmentation, occupancy, motion prediction and planning. It needs more than a detection-only nuScenes index.

Obtain complete nuScenes v1.0 trainval, map and CAN bus expansions from [nuScenes](https://www.nuscenes.org/), and occupancy labels from [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D). Follow the providers' access and license terms. Datasets are not included in this repository.

## Layout

Run from the repository root. Use symlinks if datasets live on another disk.

```text
Uni-PreFusion/
├── data/nuscenes/
│   ├── samples/
│   ├── sweeps/
│   ├── v1.0-trainval/
│   ├── maps/                       # include map expansion
│   └── can_bus/
├── occ3d/gts/<scene-name>/<sample-token>/labels.npz
├── assets/
│   ├── kmeans_motion_6.npy
│   └── kmeans_planning_6.npy
└── ckpts/
    ├── model_manifest.json
    ├── uniprefusion-final.pth
    └── hilbert_template/
        ├── curve_template_3d_rank_7.pth
        ├── curve_template_3d_rank_8.pth
        ├── curve_template_3d_rank_9.pth
        └── curve_template_3d_rank_10.pth
```

The configuration uses `data_root='data/nuscenes/'` and `occ_root='occ3d/gts'`. The latter is relative to the **repository working directory**, not automatically relative to `data_root`.

## Multitask metadata

After installing the environment/operators, use the inspected converter entry point:

```bash
export UNILION_MAMBA_BACKEND=external
python tools/data_converter/nuscenes_converter.py nuscenes \
  --root-path ./data/nuscenes \
  --canbus ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --db-save-path ./data/nuscenes \
  --extra-tag nuscenes \
  --max-sweeps 10 \
  --version v1.0
```

This creates custom `nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl` and the applicable training ground-truth database. The official full split has 28,130 training and 6,019 validation keyframes. The converter derives map, CAN bus and future-trajectory fields used by downstream tasks; a detection index from another project is not assumed compatible. Record converter revision, arguments and dataset versions.

The inherited `tools/create_data.py` has an unresolved `args.can_bus` argument mismatch; use the converter above. It imports project datasets/evaluators, so native extensions and the occupancy renderer may be required even though its output is preprocessing metadata.

## Final checkpoint and constructor assets

The checkpoint and four Hilbert tables are separate GitHub Release assets. Paths, sizes and SHA-256 values are recorded in [`ckpts/model_manifest.json`](../ckpts/model_manifest.json). The original fusion-backbone constructor loads **all four tables**; downloading only the final checkpoint is insufficient.

The small arrays under `assets/` are exact copies of the released checkpoint's anchor tensors, with provenance and hashes in the manifest:

| Array | Shape | Head |
|---|---|---|
| `kmeans_motion_6.npy` | `[10, 6, 12, 2]` | Motion |
| `kmeans_planning_6.npy` | `[3, 6, 6, 2]` | Planning |

Their constructors read these files before checkpoint loading even though corresponding parameters exist in the final state. They are not randomly regenerated substitutes. Check the arrays with `python tools/check_release.py` and all large downloads with `--hash-assets`.

Final-model evaluation does not require separate Swin, VMamba or MambaFusion initialization weights under the final public configuration. Those resources belong to staged training.

Historical generation scripts `tools/kmeans_motion.py`, `tools/kmeans_planning_6.py` and `tools/kmeans_planning_4096.py` have fixed paths/settings; the motion script also saves a plot to a fixed author path. They are not a portable replacement for the released exact arrays. The final configuration references the six-mode planning anchor, not the 4096-mode variant.
