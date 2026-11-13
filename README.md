# Score-Guided Reconstruction

Official implementation of *Potential Learning from Diffusion-Model Scores
for Implicit Manifold Reconstruction*.

The method learns a scalar implicit field `Phi` from an unorganized point cloud,
with the associated nonnegative potential `H = Phi^2 / 2`. It represents the
target codimension-one manifold by the zero level set `Phi = 0`. The
three-dimensional pipeline returns a triangular mesh; the high-dimensional
pipeline returns a zero-set point cloud.

## Get the code and data

Install Git and Git LFS, then clone the repository:

```bash
git lfs install
git clone https://github.com/zhangshiliang502-droid/score-guided-reconstruction.git
cd score-guided-reconstruction
git lfs pull
```

Run the commands below from this repository root. The complete LFS download
is about 615 MB: 374 MB of evaluation inputs and 241 MB of retained results.

## Release contents

| Component | Included |
|---|---|
| Method | Training, reconstruction and evaluation source code |
| 3D experiments | 27 configurations, selected checkpoints, meshes and metrics |
| High-dimensional experiments | Six analytic definitions, configurations and summaries |
| Paper tables | Retained SGR and SGR--DWG comparison tables |
| 3D evaluation inputs | 27 input clouds and 27 reference meshes in `datasets/`, via Git LFS |

Selected checkpoints, meshes and the 54 evaluation inputs are stored with Git
LFS. Their exact input identities are listed in
`example/lowdim/evaluation_assets.json`. Exact retraining of the selected 3D
checkpoints requires dense score support that is not included.

All 27 retained 3D experiments were trained with a budget of 100,000 updates.
The selected checkpoint step recorded in each summary may be earlier because
checkpoints were retained during the corresponding 100,000-update run.

## Data access

The exact 27 input point clouds and 27 independent reference meshes used for
the retained 3D evaluations are included under [`datasets/`](datasets/README.md).
From the repository root, download the LFS objects and verify every input:

```bash
git lfs install
git lfs pull
python tools/validate_release.py --profile lowdim-evaluation --asset-root .
```

The input bundle is 374,165,288 bytes (about 374 MB), in addition to the
retained model and reconstruction files. Original upstream sources include
the [ABC Dataset](https://deep-geometry.github.io/abc-dataset/),
the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)
and the [Open3D MonkeyModel](https://www.open3d.org/docs/release/python_api/open3d.data.MonkeyModel.html).
Exact paper-file identities and the asset-root layout are recorded in
[`example/lowdim/evaluation_assets.json`](example/lowdim/evaluation_assets.json).
Analytic reference meshes can be generated with
[`03_evaluation/generate_analytic_reference_meshes.py`](03_evaluation/generate_analytic_reference_meshes.py).
Source attribution and use of the supplied data, checkpoints and reconstructed
meshes are described in
[`ARTIFACTS.md`](ARTIFACTS.md).

## Installation

Python 3.10--3.12 and a 64-bit Linux server are supported. Full experiments
require an NVIDIA GPU and a PyTorch build compatible with the server CUDA
driver.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate CUDA-enabled PyTorch build first.
python -m pip install -r requirements.txt
python tools/validate_release.py --profile source
```

A Conda specification is also provided:

```bash
conda env create -f environment.yml
conda activate score-guided-reconstruction
```

See [`docs/TESTED_ENVIRONMENTS.md`](docs/TESTED_ENVIRONMENTS.md) for the
supported software range, CI matrix and server checks.

## Quick start

### Three-dimensional point cloud

Input is a finite numeric `N x 3` CSV or NPY array.

```bash
python 01_lowdim_sgr/code/main.py \
  --point-cloud /path/to/points.npy \
  --output-dir runs/my_3d_dataset \
  --device cuda

python 01_lowdim_sgr/code/reconstruct.py \
  --checkpoint runs/my_3d_dataset/implicit_function/best.pt \
  --point-cloud /path/to/points.npy \
  --out-mesh-npz runs/my_3d_dataset/reconstruction/zero_mesh.npz \
  --out-stats runs/my_3d_dataset/reconstruction/stats.json \
  --device cuda
```

The formal extractor uses a `360^3` grid over the input bounding box with 10%
padding on each side and retains all connected components. For large score
caches, use `01_lowdim_sgr/code/plan_gaussian_knn.py` to select KNN support.

### High-dimensional point cloud

Input is a finite numeric `N x D` NPY array with `D > 3`.

```bash
python 02_highdim_sgr/code/main.py \
  --point-cloud /path/to/points.npy \
  --output-dir runs/my_highdim_dataset \
  --device cuda

python 02_highdim_sgr/code/reconstruct.py \
  --point-cloud /path/to/points.npy \
  --checkpoint runs/my_highdim_dataset/implicit_function/best.pt \
  --output-dir runs/my_highdim_dataset/reconstruction \
  --device cuda
```

Detailed commands for evaluation and paper reproduction are provided in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Paper experiments

The six analytic high-dimensional experiments require no external geometry:

```bash
python tools/reproduce_highdim.py --examples all --dry-run
python tools/reproduce_highdim.py \
  --examples all \
  --output-root runs/paper_highdim \
  --device cuda
```

To inspect or re-evaluate the retained 3D results:

```bash
git lfs pull
python tools/validate_release.py --profile results
python tools/validate_release.py \
  --profile lowdim-evaluation \
  --asset-root .
python 03_evaluation/evaluate_lowdim_benchmark.py \
  --asset-root .
```

The supplementary small-noise comparison between empirical KNN and trained DSM
scores, which is separate from the manuscript's Tables 1--4, is documented under
[`experiments/empirical_vs_trained_small_sigma/`](experiments/empirical_vs_trained_small_sigma/).

## Repository map

| Path | Purpose |
|---|---|
| `01_lowdim_sgr/code/` | 3D score construction, implicit-field learning and mesh extraction |
| `02_highdim_sgr/code/` | High-dimensional score learning and zero-set reconstruction |
| `03_evaluation/` | Independent evaluation and table generation |
| `experiments/empirical_vs_trained_small_sigma/` | Empirical-versus-trained score ablation |
| `example/` | Paper configurations, retained outputs and analytic definitions |
| `tools/` | Validation, reproduction and result-packaging utilities |
| `docs/` | Reproducibility, data, environment and release validation |

## Documentation

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): supported claims and
  executable paper workflows.
- [`docs/PAPER_ALIGNMENT.md`](docs/PAPER_ALIGNMENT.md): manuscript-to-code mapping,
  numerical checks and verification limits.
- [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md): evaluation inputs, provenance and
  retained-result packaging.
- [`docs/TESTED_ENVIRONMENTS.md`](docs/TESTED_ENVIRONMENTS.md): Linux server and
  CI environments.
- [`docs/VALIDATION.md`](docs/VALIDATION.md): validation commands and coverage.
- [`01_lowdim_sgr/README.md`](01_lowdim_sgr/README.md) and
  [`02_highdim_sgr/README.md`](02_highdim_sgr/README.md): implementation notes.

Seeds and protocol parameters are recorded, but bit-for-bit equality is not
guaranteed across GPU, CUDA, PyTorch or scientific-library versions.

## Citation

GitHub-compatible citation metadata are provided in [`CITATION.cff`](CITATION.cff).
The software release can also be cited as:

```bibtex
@software{zhang2026scoreguided,
  author  = {Shiliang Zhang},
  title   = {Score-Guided Reconstruction},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/zhangshiliang502-droid/score-guided-reconstruction}
}
```

## License

Repository-authored source code, documentation, configurations and analytic
definitions are released under the [MIT License](LICENSE). Retained
checkpoints, reconstructed meshes and evaluation artifacts are research
artifacts and are not granted a separate MIT license. Their provenance and
applicable upstream terms are described in [`ARTIFACTS.md`](ARTIFACTS.md).
