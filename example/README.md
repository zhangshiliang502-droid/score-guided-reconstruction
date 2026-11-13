# Paper experiments

This directory contains frozen experiment records. Executable method code
remains in the numbered source directories at the repository root.

## Three-dimensional results

The 27 selected examples follow one layout:

```text
example_<dataset>/
├── config/training.json
├── dataset/reference.json
└── result/
    ├── summary.json
    ├── evaluation.json
    ├── implicit_function/best.pt
    └── reconstruction/{zero_mesh.npz,stats.json}
```

Configurations record normalization, score support, optimization and mesh
extraction. Reference records identify independent evaluation geometry.
Checkpoints, meshes and the 54 input clouds and reference meshes are
distributed through Git LFS. The inputs are stored under
[`datasets/`](../datasets/README.md); use the repository root as `--asset-root`.

Aggregate SGR--DWG results are under `lowdim/summary/`.

## High-dimensional results

The six analytic examples contain:

```text
example_<D>d/
├── analytic_expression.py
├── config/training.json
└── result/summary.json
```

Preview or execute a configuration with:

```bash
python tools/reproduce_highdim.py --examples example_6d --dry-run
python tools/reproduce_highdim.py --examples example_6d --device cuda
```

See [`lowdim/README.md`](lowdim/README.md) and
[`highdim/README.md`](highdim/README.md) for record-specific details.
