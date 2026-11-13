# Data availability

This document describes the data and artifacts associated with *Potential
Learning from Diffusion-Model Scores for Implicit Manifold Reconstruction*.
The exact three-dimensional evaluation inputs are included through Git LFS;
source provenance and applicable upstream terms remain attached to the data.

## Availability summary

| Resource | Included | Access | Verification |
|---|---|---|---|
| Source code and 33 experiment configurations | Yes | Ordinary Git clone | `validate_release.py --profile source` |
| Selected 3D checkpoints and meshes | Yes | Git LFS | `validate_release.py --profile results` |
| 27 3D input clouds and 27 independent reference meshes | Yes | `datasets/`, through Git LFS | `--profile lowdim-evaluation --asset-root .` |
| Dense score support for selected 3D models | No | Not distributed with this release | Exact selected-model retraining is unavailable without it |
| Six analytic high-dimensional datasets | Generated on demand | Included analytic definitions and configurations | `--profile highdim-data` |

## Download and validate the 3D inputs

From the repository root, retrieve all retained outputs and evaluation inputs:

```bash
git lfs install
git lfs pull
python tools/validate_release.py --profile lowdim-evaluation --asset-root .
```

The input bundle contains exactly 54 files, totaling 374,165,288 bytes (about
374 MB), in addition to the retained checkpoints and reconstructions. A source
clone without LFS downloads contains pointer text in place of the data files;
the low-dimensional evaluation validator rejects these unresolved pointers.

Each of the 27 examples uses one complete input cloud and one independent
reference mesh. The original relative paths, byte counts and SHA-256 digests
are preserved in `example/lowdim/evaluation_assets.json`. A dataset-by-dataset
file index is available in [`datasets/README.md`](../datasets/README.md).

The repository root is the asset root for the bundled data:

```text
<repository-root>/
└── datasets/
    ├── Bunny/
    ├── Knot/
    ├── Armadillo/
    ├── Monkey/
    ├── Sphere/
    ├── FlowerPetal/
    ├── CV2Radial_a020_f16/
    ├── ABC_Official_Selected_8/
    ├── ABC_Official_Additional_12/
    └── source_meshes/
```

For example, re-evaluate all retained SGR meshes with:

```bash
python 03_evaluation/evaluate_lowdim_benchmark.py --asset-root .
```

An existing external asset directory can still be supplied with
`--asset-root /path/to/assets` if it contains the same `datasets/` layout and
matches all 54 manifest hashes. Upstream geometry with different components
or independently resampled points is not interchangeable with the frozen
paper inputs.

## Source provenance and applicable terms

| Group | Source | Included paper inputs | Applicable terms |
|---|---|---|---|
| ABC | [ABC Dataset v00](https://deep-geometry.github.io/abc-dataset/) | 20 selected reference components and 20 author-generated point clouds | Follow the ABC license notice and [Onshape Terms of Use](https://www.onshape.com/en/legal/terms-of-use), and cite the ABC dataset |
| Bunny and Armadillo | [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) | Two reference meshes and two author-generated point clouds | Follow the usage conditions on the official source page and acknowledge the source |
| Monkey | [Open3D MonkeyModel](https://www.open3d.org/docs/release/python_api/open3d.data.MonkeyModel.html) | One reference mesh and one author-generated point cloud | Follow the terms supplied with the upstream asset |
| Sphere, CV2Radial and FlowerPetal | [`generate_analytic_reference_meshes.py`](../03_evaluation/generate_analytic_reference_meshes.py) | Three analytic reference meshes and three exact training clouds | Repository-authored analytic code is covered by the repository license |
| Knot | Author-supplied geometry | One reference mesh and one exact input cloud | This repository does not assert additional third-party rights |

The data are research inputs. Their inclusion does not extend the source-code
MIT license to third-party geometry; see [`ARTIFACTS.md`](../ARTIFACTS.md).
The manifest records the source group for every file. To regenerate the
identity record from the bundled inputs, run:

```bash
python tools/build_lowdim_asset_manifest.py --asset-root .
```

The analytic reference meshes for Sphere, CV2Radial and FlowerPetal are also
regenerable from the included source definitions. For exact paper evaluation,
use the hash-verified versions already supplied through Git LFS.

## Dense 3D score support

The generic 3D workflow constructs a Gaussian-KNN score cache from a supplied
point cloud. The selected paper checkpoints instead used dense external score
support. Those caches are not included in this repository or in the 54-file
evaluation manifest.

A compatible cache can be passed with `main.py --score-cache-pt`. Without the
original cache or score-support samples, the release supports new-data
training and exact retained-result evaluation, but not exact retraining of a
selected 3D model.

## Analytic high-dimensional data

The six high-dimensional datasets are generated from the included analytic
definitions and require no external geometry:

```bash
python tools/reproduce_highdim.py --examples all --stages generate
python tools/validate_release.py --profile highdim-data
```

Generated arrays and metadata are excluded from Git and can be recreated from
their frozen configurations and seeds.
