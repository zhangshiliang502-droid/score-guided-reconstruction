# Manuscript and repository correspondence

The 2026-11-13 audit compared this repository with the 2026-11-12 LaTeX
snapshot of *Potential Learning from Diffusion-Model Scores for Implicit
Manifold Reconstruction* by Shiliang Zhang, Yiming Xu and Jiayu Zhai.
The checked manuscript source has SHA-256
`6d8beefed50d721a7124eaf99f7a858ba9d7c92639ad69a2b6e24d0f8998fcc2`.

## Method and protocol

| Manuscript item | Implementation and interpretation |
|---|---|
| Implicit representation | The network outputs `Phi`. The associated nonnegative potential is `H = Phi^2 / 2`; both have the same zero set. |
| Mean/max-radius normalization | Both pipelines center the input by its empirical mean and divide by its maximum centered radius. Re-evaluation uses the saved training mean and scale. |
| Local sampling | A uniform input anchor and a uniform unit direction are combined with `r = max(2.5*sigma + sigma*xi, 0)`, `xi ~ N(0,1)`. This is a Gaussian radius clamped at zero. |
| Global sampling | Uniform sampling in the normalized input bounding box with 30% of each coordinate span added on each side. |
| Empirical score | `01_lowdim_sgr/code/construct_score.py` computes the Gaussian-weighted neighbor mean minus the query, divided by `sigma^2`. Its generic support is the input cloud; selected paper models used additional dense support. |
| Conservative score | `02_highdim_sgr/code/construct_score.py` learns `U = sigma*E`. The score `-grad(U)/sigma` equals the manuscript's `-grad(E)`. |
| Score-pull loss | `1000 * mean(||Phi*grad(Phi)/||grad(Phi)|| + sigma^2*score||^2)`. Small gradient norms are bounded below for numerical stability. |
| Exterior direction loss | `lambda_dir * mean(1-cos(-grad(Phi^2), score))`. The 3D per-dataset weight is recorded as `lambda_global`; all six high-dimensional configurations use 20. |
| SIREN | Four hidden layers, width 128, `omega0=20`, one scalar output. The 3D analytic chain-rule gradient is mathematically equivalent to automatic differentiation, including parameter derivatives. |
| 3D extraction | Marching cubes on a `360^3` grid, 10% input-box padding per side, all components retained. |
| High-dimensional extraction | One million uniform box candidates, 20 bounded Newton steps, maximum step 0.1, squared-gradient denominator bounded below by `1e-12`, box clipping, acceptance at `abs(Phi)<=1e-4` and gradient norm `>1e-6`, 250,000 retained points. |
| 3D distance metrics | One million independent area-weighted samples per mesh. Nearest sample distances approximate point-to-surface RMSE and define the squared bidirectional Chamfer estimate. |
| 3D ghosts | A component is remote when its median vertex distance to the full input cloud exceeds three times the full cloud's median eighth-neighbor spacing; the reported ratio is area-weighted. |
| High-dimensional evaluation | Exactly 100,000 converged input projections; spacing from 250,000 reference points; pointwise ghosts use a strict distance threshold of three times the median eighth-neighbor spacing. |

`clamped_gaussian` is the descriptive sampling name in new caches and the
configuration records. Retained binary checkpoints preserve their historical
`gaussian_abs` label. The original dense-cache generator clamps negative radii
to zero, so this label does not denote taking an absolute value. Version-4
readers accept the legacy label only with the explicit `clamp_zero` policy
and the remaining required sampling metadata.

## Recorded numerical results

All 342 numeric cells in manuscript Tables 1--4 agree with the corresponding
repository records at the printed precision:

| Table | Checked records | Numeric cells |
|---|---|---:|
| 1: RMSE | 27 rows, both methods and DWG/SGR ratios | 81 |
| 2: squared Chamfer | 27 rows, both methods and DWG/SGR ratios | 81 |
| 3: ghost components | 27 rows, both methods' component counts, ghost counts and area percentages | 162 |
| 4: high-dimensional results | Six rows, RMSE, squared Chamfer and ghost ratio | 18 |

The 27 individual SGR evaluation records and summaries also agree with the
aggregate table. Their checkpoint metadata matches the recorded architecture,
normalization, training seed, selected step, loss weight, score bandwidth and
KNN size. SGR wins 24 RMSE comparisons and 23 Chamfer comparisons; median
DWG/SGR ratios round to 1.270 and 1.609. All SGR ghost counts are zero.
DWG ghost-area percentages round to 4.679% for CV2Radial and 18.386% for Sphere.

The aggregate records are under
[`example/lowdim/summary/unified_bbox10/`](../example/lowdim/summary/unified_bbox10/).
The six high-dimensional configurations and summaries are under
[`example/highdim/`](../example/highdim/README.md).

## File names and supplementary experiments

The numbered source directories describe pipeline stages; `lowdim` refers to
the 3D ambient-space benchmark. Benchmark entry points use these names:

- `01_lowdim_sgr/code/reconstruct_lowdim_benchmark.py`: extract all 27 selected
  SGR models using their recorded domains.
- `03_evaluation/evaluate_lowdim_benchmark.py`: re-evaluate all 27 retained
  SGR meshes and refresh their individual records.
- `03_evaluation/evaluate_lowdim_comparison.py`: compare independently
  extracted SGR and DWG results and write the aggregate tables.

Dataset identifiers, source mesh names and schema versions preserve their
recorded identities. The small-noise score comparison under
[`experiments/empirical_vs_trained_small_sigma/`](../experiments/empirical_vs_trained_small_sigma/README.md)
is supplementary and separate from manuscript Tables 1--4. Its sphere table
is named `table_sphere_score_comparison.tex`.

## Verification scope

The audit checked manuscript values, source formulas, configuration records,
retained 3D checkpoint metadata, and the SHA-256 identities of all 54 bundled
evaluation inputs. Formula tests independently exercise the Gaussian
log-density gradient, analytic SIREN derivatives, loss sign and gradient
normalization, radius clamping, energy scaling, and Newton acceptance rules.
The reduced CPU training smoke test checks the generic end-to-end interface.

This audit did not perform new full GPU training or independently recompute
all benchmark metrics. Selected 3D models used version-2 dense-support score
caches that are not bundled; the generic trainer accepts version-4 input-cloud
caches. Table 4 has retained summaries and runnable configurations, but its
matching trained models and reconstructed point sets are not bundled, so its
published runs cannot be independently re-evaluated from this download alone.
DWG meshes and run logs, full parameter-search records, and manuscript figures
are also outside the supplied material. These omissions do not change the
recorded table agreement.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for executable workflows and
[`VALIDATION.md`](VALIDATION.md) for the validation coverage.
