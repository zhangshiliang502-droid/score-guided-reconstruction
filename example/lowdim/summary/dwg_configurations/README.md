# DWG configurations for the unified 10% comparison

[`configurations.csv`](configurations.csv) records the selected DWG settings
used in the comparison. Each descriptive configuration identifier refers to
the complete parameter row for one retained reconstruction.

The identifier convention is:

```text
dwg-<initialization>-<sample-count>-d<octree-depth>[-<distinguishing-option>]
```

`gm` denotes Gauss-map initialization. `random` denotes random
initialization. The suffixes `l10` and `l50` distinguish the recorded lambda
parameter, while `area` denotes enabled area-weight computation.

## Field definitions

| Field | Meaning |
|---|---|
| `dataset` | Evaluation dataset |
| `configuration_id` | Public identifier for the complete parameter row |
| `sample_count` | Number of input samples supplied to DWG |
| `initialization` | Human-readable initialization mode |
| `input_normal_mode` | DWG `--in_normal` value (`0`: random, `3`: Gauss map) |
| `max_octree_depth` | DWG `--max_depth` value |
| `iteration_grid_resolution` | Marching-cubes resolution used during iteration |
| `final_grid_resolution` | Resolution of the retained extraction |
| `iterations` | Number of DWG iterations |
| `use_radii` | Whether DWG screening radii are enabled |
| `knn_neighbors` | Main neighborhood size |
| `final_knn_neighbors` | Neighborhood size for final extraction |
| `normal_knn_neighbors` | Neighborhood size used for normals |
| `lambda_parameter` | Lambda value recorded by the experiment wrapper |
| `compute_area_weights` | Whether DWG area weights are computed |
| `output_mode` | DWG output mode (`1`: mesh) |
| `bbox_padding_fraction_per_side` | Fractional padding on each bounding-box side |

All retained meshes use a final resolution of 360 and 10% padding on each
side of the corresponding input bounding box. DWG is an external comparison
method and is not implemented in this repository.
