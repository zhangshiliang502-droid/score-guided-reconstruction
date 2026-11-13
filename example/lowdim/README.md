# Three-dimensional paper results

This directory contains 27 selected SGR results: 20 ABC geometries and seven
named or synthetic surfaces.

| Record | Contents |
|---|---|
| `config/training.json` | Data identity, normalization, score support, loss, seed and extraction protocol |
| `dataset/reference.json` | Independent reference mesh, transform and evaluation offset |
| `result/implicit_function/best.pt` | Selected SIREN checkpoint |
| `result/reconstruction/zero_mesh.npz` | Complete `360^3`, 10%-padded zero mesh |
| `result/reconstruction/stats.json` | Extraction domain and mesh statistics |
| `result/evaluation.json` | Full metric record |
| `result/summary.json` | Compact paper result |

Every retained experiment uses a 100,000-update training budget. The selected
checkpoint may precede update 100,000 because checkpoints were retained during
the corresponding run.

Checkpoints, meshes and the 54 required input and reference files are
distributed through Git LFS. The inputs are stored under
[`datasets/`](../../datasets/README.md) and are identified in
`evaluation_assets.json` by relative path, size and SHA-256. After
`git lfs pull`, validate them from the repository root with:

```bash
python tools/validate_release.py --profile lowdim-evaluation --asset-root .
```

The manifest supports exact re-evaluation; it does not include the dense score
support required for exact selected-model retraining.

Aggregate SGR--DWG results are in `summary/unified_bbox10/`, with selected DWG
settings in `summary/dwg_configurations/`.
