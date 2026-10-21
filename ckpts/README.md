# Model assets

Large model files are hosted in the GitHub release rather than Git history.
Run from the repository root:

```bash
python tools/download_models.py
python tools/download_models.py --verify-only
```

The expected destinations are `ckpts/uniprefusion-final.pth` and four lookup
tables under `ckpts/hilbert_template/`. Exact sizes, SHA-256 digests, and public
download URLs are pinned in [`model_manifest.json`](model_manifest.json).
See [`docs/checkpoints.md`](../docs/checkpoints.md) for manual downloads,
resuming interrupted transfers, and constructor anchor provenance.
