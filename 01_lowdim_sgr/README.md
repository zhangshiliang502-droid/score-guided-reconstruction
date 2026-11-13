# Three-dimensional SGR

This module reconstructs a codimension-one surface from an unorganized
`N x 3` point cloud. It estimates a Gaussian-smoothed score, learns a scalar
SIREN implicit field `Phi`, and extracts the zero level set. The paper's
nonnegative potential is `H = Phi^2 / 2`.

## Pipeline

```text
point cloud
├── plan_gaussian_knn.py  optional KNN support planning
└── main.py
    ├── construct_score.py  Gaussian-KNN score cache
    └── train_phi.py        implicit_function/best.pt

best.pt + point cloud + reconstruct.py  ->  zero_mesh.npz
```

Coordinates are centered by the input mean and divided by the maximum centered
radius. Local queries use a uniform input anchor, a uniform unit direction,
and radius `max(2.5*sigma + sigma*xi, 0)` for `xi ~ N(0,1)`. This sampling
law is named `clamped_gaussian`. The legacy `gaussian_abs` label in retained
checkpoint metadata denotes the same clamping operation; it does not take
the absolute value of a Gaussian draw. Global queries are drawn
from the normalized bounding box with 30% padding. `models.py` evaluates the
SIREN input gradient analytically.

Before constructing a large score cache, estimate K values for 98% and 99%
Gaussian kernel mass:

```bash
python 01_lowdim_sgr/code/plan_gaussian_knn.py \
  --point-cloud /path/to/points.npy \
  --num-samples 40000000 \
  --sigma 0.002 \
  --json-out runs/gaussian_knn_plan.json
```

`reconstruct.py` evaluates the field on a `360^3` grid over the physical input
bounding box with 10% padding on each side. Marching cubes extracts level zero;
all connected components are retained.

The generic workflow builds its score cache from the supplied point cloud.
The generic `main.py --score-cache-pt` interface accepts validated version-4
input-cloud caches. The selected paper checkpoints record older dense-support
caches, which are not included and are not directly accepted by that interface.
The bundled workflow therefore supports re-evaluation of selected paper models,
while exact selected-model retraining needs the original dense support and a
matching cache contract.

`reconstruct_lowdim_benchmark.py` runs extraction for all 27 selected checkpoints
using their recorded `360^3`, 10%-padded domains.

See the root [`README.md`](../README.md) for commands and
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for supported claims.
