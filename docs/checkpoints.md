# Checkpoints and required assets

The trained Uni-PreFusion checkpoint and Hilbert lookup tables are distributed as
five separate assets in the [`v1.0.0` release](https://github.com/FangWD520/Uni-PreFusion/releases/tag/v1.0.0).
Their exact file sizes and SHA-256 digests are pinned in
[`ckpts/model_manifest.json`](../ckpts/model_manifest.json). Each individual file
is smaller than 2 GiB. The complete download is **2,210,187,821 bytes**
(approximately 2.06 GiB).

From the repository root, run:

```bash
python tools/download_models.py
python tools/download_models.py --verify-only
```

The downloader uses only the Python standard library. It verifies both the file
size and SHA-256 before creating a final filename, and skips already verified
assets. It downloads all five assets by default. `--verify-only` makes no network
requests and returns a nonzero exit code if any required asset is missing or
mismatched.

| Release asset | Local destination | Bytes |
| --- | --- | ---: |
| `uniprefusion-final.pth` | `ckpts/uniprefusion-final.pth` | 977,058,870 |
| `curve_template_3d_rank_7.pth` | `ckpts/hilbert_template/curve_template_3d_rank_7.pth` | 16,777,963 |
| `curve_template_3d_rank_8.pth` | `ckpts/hilbert_template/curve_template_3d_rank_8.pth` | 134,218,475 |
| `curve_template_3d_rank_9.pth` | `ckpts/hilbert_template/curve_template_3d_rank_9.pth` | 1,073,742,571 |
| `curve_template_3d_rank_10.pth` | `ckpts/hilbert_template/curve_template_3d_rank_10.pth` | 8,389,942 |

You may instead download the release assets manually, place them at the listed
destinations, and run the verification command. Do not place these large files
in Git history.

## Interrupted downloads

Re-run the same command after a network interruption. Partial downloads use
`<filename>.part` together with a matching `<filename>.part.json` ownership
record. A resumed response must contain the exact expected `Content-Range`.
If the server ignores the range request and returns a complete file, the tool
restarts only its own identified partial file. Existing final files and foreign
partial files are never overwritten.

A size or checksum mismatch stops the download and preserves the files for
inspection. After confirming that no downloader is running, move the affected
bad file out of the destination before retrying. For a bad partial download,
move both its `.part` file and `.part.json` record together. A `.download.lock`
file normally disappears on exit; remove a stale lock only after checking that
the previous process has exited.

## Constructor anchors

The small constructor arrays are included in Git:

* `assets/kmeans_motion_6.npy`: shape `(10, 6, 12, 2)`, float32.
* `assets/kmeans_planning_6.npy`: shape `(3, 6, 6, 2)`, float32.

These arrays match the motion and planning anchor tensors in the released
checkpoint exactly. Their file and raw tensor hashes are recorded in the asset
manifest. They allow the original model constructor to initialize without
requiring private training-data directories; including them does not indicate
that motion prediction or planning is evaluated in the detection results.

The release checkpoint retains its original serialized parameter names to
support strict loading. The repository's public model name is Uni-PreFusion;
inherited internal module names are retained where changing them would break
checkpoint compatibility. See the main README for installation, configuration,
evaluation protocol, and the distinction between manuscript results and later
reproduction checks.
