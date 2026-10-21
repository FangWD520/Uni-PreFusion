# Reproducibility scope

This release preserves the supplied Uni-PreFusion architecture and parameter/module names. Public configuration aliases inherit the original configurations and adjust documentation-facing names, logs and exact constructor-anchor paths. They do not silently substitute experimental numerical fixes or acceleration methods.

## Release checks

`tools/check_release.py` checks required source paths, Python syntax, configuration inheritance and small anchor hashes. Optional arguments check downloaded checkpoint/table sizes and SHA-256 values, or explicitly import installed native extensions. None of these checks executes a model forward or independently reproduces manuscript metrics.

The public packaging has not been newly accepted through a clean-machine build, complete five-task inference and every official task evaluation. Manuscript experiments, asset integrity and a new reproduction run are separate pieces of evidence.

## Known numerical consistency limitation

Investigation of the original frontend has observed cross-process numerical differences on fixed inputs and weights. In a bounded 32-frame study, repeated direct-loading runs agreed, while preloaded-input runs included samples beyond `atol=rtol=1e-5`; differences could propagate to candidate labels and decoded detections. This does not establish preloading itself as the cause or identify one responsible operator.

The supplied VMamba code also sets cuDNN policy at import time, including `benchmark=True` and `deterministic=True`. Read actual runtime settings after the complete model is imported/built; pre-import requests can be overwritten. Disabling benchmarking alone has not established full cross-process consistency. This release includes no silent experimental fix, relaxed tolerance or bitwise-reproducibility claim.

For comparisons, fix weights/configuration, inputs, sequence boundaries, GPU, precision and operator builds. Establish a stability range using independent processes before attributing small accuracy or latency changes to a method. A seed or `--deterministic` flag alone is not proof that every custom/native operator is deterministic.

## Record with each result

- Repository revision, public configuration and command-line overrides.
- Checkpoint, four Hilbert tables and anchor SHA-256 values from `ckpts/model_manifest.json`.
- nuScenes/Occ3D versions, converter revision and metadata hashes.
- Python, PyTorch, toolkit/driver, MMCV, MMDetection3D, spconv/cumm, Mamba and custom-extension versions/origins.
- Selected Mamba backend, actual cuDNN/TF32 policy, GPU architecture, sequence reset rules and inference precision.
- Split, frame count, task switches and complete metric outputs.

The final public configuration retains six cameras, ten point-cloud sweeps, all five tasks and original rear diffusion ratio 0.2. Registry name `UniLION` remains for checkpoint compatibility. `UNILION_MAMBA_BACKEND=external` preserves the supplied trained structure; selecting the original bidirectional backend is a different configuration.

The new installation guide uses CUDA 12.4-compatible spconv wheels rather than historical CUDA 11.6 entries. This explicit packaging choice is not evidence of unchanged numerical results between environments. Validate checkpoint compatibility and task metrics after environment changes.

## Evaluation and retraining

Final weights and exact constructor assets support the final evaluation workflow after environment/data preparation. They do not include every original initialization checkpoint, optimizer state, stage-selection record or training log. The [training guide](training.md) describes the dependencies and required explicit inputs without claiming an independently rerun historical training reproduction.

Dataset terms and third-party code licenses remain separate from checkpoint availability. Consult repository license and attribution files before redistribution or other uses.
