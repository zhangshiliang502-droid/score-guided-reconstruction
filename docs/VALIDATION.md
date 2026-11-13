# Release validation

This guide describes the supplied material and the checks available to verify
it. Reproducibility scope and data provenance are documented
in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and
[`DATA_LAYOUT.md`](DATA_LAYOUT.md).

## Included release material

| Material | Location |
|---|---|
| Source code, configurations and analytic definitions | Ordinary Git files |
| 27 selected 3D checkpoints and 27 reconstructed meshes | Git LFS |
| 27 exact input point clouds and 27 independent reference meshes | `datasets/`, through Git LFS |
| Six high-dimensional analytic experiment definitions | `example/highdim/` |
| Evaluation records and publication tables | `example/lowdim/` and `example/highdim/` |
| Software license | `LICENSE` |
| Artifact provenance and use notice | `ARTIFACTS.md` |
| Citation metadata | `CITATION.cff` |

The 54 manifest-identified inputs retain their source provenance and upstream
terms; see [`ARTIFACTS.md`](../ARTIFACTS.md). Dense score caches, run directories,
logs and machine-specific files are not part of this input bundle.

## Validation profiles

```bash
python tools/validate_release.py --profile source
python tools/validate_release.py --profile results
python tools/validate_release.py \
  --profile lowdim-evaluation \
  --asset-root .
python tools/reproduce_highdim.py --examples all --dry-run
python -m unittest discover -s tests -v
python tools/smoke_test_lowdim.py
```

These checks cover the method and experiment source syntax, the 27
low-dimensional and six high-dimensional experiment records, Git LFS policy, retained checkpoint
metadata, mesh structure, portable paths and the 54-file input manifest. The
low-dimensional evaluation profile additionally verifies every input file by
size and SHA-256.

`tests/test_paper_formulas.py` checks score and energy conventions, the exact
SIREN chain-rule derivatives, the local/global loss, clamped local sampling,
legacy sampling-label validation, and bounded Newton projection. The separate
manuscript comparison and its verification limits are documented in
[`PAPER_ALIGNMENT.md`](PAPER_ALIGNMENT.md).

## Data paths

All 3D evaluation inputs are resolved relative to `--asset-root` under
`datasets/`. The bundled inputs use `--asset-root .` from the repository root.
Generated runs use caller-selected output directories, and all
stored experiment paths follow the documented relative-path contract.

## Reproducibility boundary

The release supports fitting new point clouds, inspecting all retained 3D
outputs, verifying retained results against the bundled evaluation inputs and
regenerating the analytic high-dimensional datasets. The original dense score
caches used for the selected 3D checkpoints are not distributed, so exact
selected-checkpoint retraining is outside the bundled artifact scope.
