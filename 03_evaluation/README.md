# Evaluation

Evaluation is independent of training and does not modify model checkpoints.

## Three-dimensional metrics

`evaluate_lowdim_reconstruction.py` reports:

- point-to-surface RMSE from one million independent reference samples;
- squared bidirectional Chamfer distance from one million samples per surface;
- ghost-component count and area ratio on the complete extracted mesh.

The ghost threshold is three times the median eighth-nearest-neighbor spacing
of the complete input cloud. All distances use the saved training
normalization.

## High-dimensional metrics

`evaluate_highdim_reconstruction.py` projects 100,000 input points and evaluates
250,000 retained reconstruction points in the formal paper protocol. A point
is a ghost when its reference distance is strictly greater than three times
the median eighth-nearest-neighbor spacing of the configured reference sample.

The evaluator rejects nonconverged projections and checkpoints whose recorded
architecture, loss or projection settings disagree with the configuration.

## Entry points

| Script | Purpose |
|---|---|
| `evaluate_lowdim_reconstruction.py` | Evaluate one 3D mesh |
| `evaluate_lowdim_benchmark.py` | Re-evaluate all 27 retained SGR meshes |
| `evaluate_lowdim_comparison.py` | Compare SGR and DWG |
| `generate_analytic_reference_meshes.py` | Generate independent analytic 3D references |
| `evaluate_highdim_reconstruction.py` | Evaluate a high-dimensional zero set |

Executable paper workflows are in
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).
