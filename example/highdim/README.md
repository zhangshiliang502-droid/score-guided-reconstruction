# Analytic high-dimensional experiments

The benchmark contains six codimension-one manifolds:

| Example | Ambient / intrinsic dimension | Topology |
|---|---:|---|
| `example_4d` | 4 / 3 | `S^1 x S^2` |
| `example_6d` | 6 / 5 | `T^2 x S^3` |
| `example_8d` | 8 / 7 | `T^3 x S^4` |
| `example_9d` | 9 / 8 | `T^4 x S^4` |
| `example_10d` | 10 / 9 | `S^4 x S^5` |
| `example_12d` | 12 / 11 | `S^5 x S^6` |

Each directory contains the analytic definition, a complete training and
evaluation configuration, and a compact result summary. Analytic expressions
are used only for data generation and final evaluation; model fitting receives
sampled point clouds only.

```bash
python tools/reproduce_highdim.py --examples example_4d --dry-run
python tools/reproduce_highdim.py --examples example_4d --device cuda
python tools/reproduce_highdim.py --examples all --device cuda
```

Generated arrays and run artifacts are excluded from Git. Completed stages are
reused unless `--force` is specified.
