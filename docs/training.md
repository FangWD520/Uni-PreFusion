# Evaluation and staged training

Complete [installation](INSTALL.md) and [data preparation](data_preparation.md) first. Commands run from the repository root, with the source paths in the installation guide and `UNILION_MAMBA_BACKEND=external`.

## Final evaluation

```bash
export UNILION_MAMBA_BACKEND=external
python tools/check_release.py --hash-assets
python tools/test.py configs/uniprefusion_seq_e2e.py \
  ckpts/uniprefusion-final.pth --eval mAP
```

For distributed evaluation, argument order is **configuration, checkpoint, GPU count**:

```bash
bash tools/dist_test.sh configs/uniprefusion_seq_e2e.py \
  ckpts/uniprefusion-final.pth 8 --eval mAP
```

The final configuration enables all five tasks. The inherited `--eval mAP` enters dataset evaluation, while task switches select the other task evaluators. Tracking evaluation is disabled. Retain complete evaluation output and the actual configuration with any metric. Loading the final checkpoint into a reduced training-stage configuration does not establish a strict match because task-head keys can differ.

## Stage dependencies

| Stage | Public configuration | Tasks |
|---|---|---|
| Detection | `configs/uniprefusion_det.py` | Detection |
| Detection + map | `configs/uniprefusion_det_map.py` | Detection, map |
| Temporal perception | `configs/uniprefusion_seq_perception.py` | Detection, map, occupancy |
| Final joint stage | `configs/uniprefusion_seq_e2e.py` | Detection, map, occupancy, motion, planning |

Aliases preserve original stage structures, model types and state keys. They adjust output/log names and final anchor paths. Default logging uses text and TensorBoard without enabling WandB uploads.

**The final-model download is not a complete staged-training reproduction package.** Exact initialization checkpoints and original stage-selection/training records are required to reproduce the historical training path. `load_from=None` is intentional: explicitly identify initialization rather than silently loading an author's local checkpoint. If those inputs are unavailable, the following commands describe dependencies and do not establish a complete paper retraining reproduction.

### Detector initialization

Obtain the correct original UniLION and MambaFusion checkpoints under their terms and record their hashes:

```bash
python tools/merge_mambafusion_unilion_ckpts.py \
  --config configs/uniprefusion_det.py \
  --unilion ckpts/unilion_swin_384_seq_e2e.pth \
  --mambafusion ckpts/checkpoint_epoch_10.pth \
  --out ckpts/uniprefusion_det_init.pth
python tools/audit_mambafusion_unilion_init.py \
  ckpts/uniprefusion_det_init.pth.report.json --strict-mm
bash tools/dist_train.sh configs/uniprefusion_det.py 8 \
  --cfg-options load_from=ckpts/uniprefusion_det_init.pth
```

### Transfer between stages

Each transfer combines the selected previous-stage result with appropriate additional components from the source UniLION checkpoint. Select and record exact previous checkpoints; filenames below illustrate the dependency and do not refer to additional released assets.

```bash
python tools/merge_det_to_det_map_init.py \
  --config configs/uniprefusion_det_map.py \
  --det outputs/Uni-PreFusion/det/latest.pth \
  --unilion ckpts/unilion_swin_384_seq_e2e.pth \
  --out ckpts/uniprefusion_det_map_init.pth
bash tools/dist_train.sh configs/uniprefusion_det_map.py 8 \
  --cfg-options load_from=ckpts/uniprefusion_det_map_init.pth

python tools/merge_det_map_to_seq_perception_init.py \
  --config configs/uniprefusion_seq_perception.py \
  --det-map outputs/Uni-PreFusion/det_map/latest.pth \
  --unilion ckpts/unilion_swin_384_seq_e2e.pth \
  --out ckpts/uniprefusion_seq_perception_init.pth
bash tools/dist_train.sh configs/uniprefusion_seq_perception.py 8 \
  --cfg-options load_from=ckpts/uniprefusion_seq_perception_init.pth

python tools/merge_seq_perception_to_seq_e2e_init.py \
  --config configs/uniprefusion_seq_e2e.py \
  --seq-perception outputs/Uni-PreFusion/seq_perception/latest.pth \
  --unilion ckpts/unilion_swin_384_seq_e2e.pth \
  --out ckpts/uniprefusion_seq_e2e_init.pth
bash tools/dist_train.sh configs/uniprefusion_seq_e2e.py 8 \
  --cfg-options load_from=ckpts/uniprefusion_seq_e2e_init.pth
```

Inherited schedules assume configured GPU count, batch size and iteration accounting. Changing GPU count without checking total batch size, learning rate and effective epochs changes the experiment. Save initialization/output hashes, config dumps, seeds, resolved versions, GPU, precision and training budget. Resume an interrupted stage with its own checkpoint using `--resume-from`.
