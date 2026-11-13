# High-dimensional Energy-SGR

This module reconstructs a codimension-one hypersurface from an `N x D` NPY
array with `D > 3`.

```text
point cloud + metadata
└── main.py
    ├── construct_score.py  conservative Energy checkpoint
    └── train_phi.py        implicit SIREN checkpoint

best.pt + reconstruct.py  ->  normalized and physical zero-set points
```

`construct_score.py` trains an EnergyMLP at `sigma=0.01`. The network outputs
the scaled energy `U = sigma*E`; the conservative score is
`-grad(U)/sigma = -grad(E)`, using the paper's energy convention.
`train_phi.py` then fits a four-hidden-layer,
128-unit SIREN with `omega0=20` using local score-pull and global direction
losses. Its output is the implicit field `Phi`, and the associated
nonnegative potential is `H = Phi^2 / 2`.

`reconstruct.py` samples the normalized bounding box and uses the bounded
Newton update in `projection.py` to project candidates to `Phi=0`. Points that
fail the residual or gradient criteria are rejected.

Paper data generation is isolated in `generate_paper_data.py`. Analytic
expressions are used only by the explicit generator and evaluator; production
training and reconstruction receive sampled point clouds only.

Preview a paper workflow without running it:

```bash
python tools/reproduce_highdim.py --examples example_6d --dry-run
```

See [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the complete
workflow and [`example/highdim/README.md`](../example/highdim/README.md) for the
six analytic geometries.
