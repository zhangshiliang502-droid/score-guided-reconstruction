# Exact three-dimensional paper inputs

This directory contains the 27 input point clouds and 27 independent reference
meshes used to evaluate the retained paper results. All 54 data files are stored
through Git LFS, with a total uncompressed size of 374,165,288 bytes (about 374 MB).

From the repository root:

```bash
git lfs install
git lfs pull
python tools/validate_release.py --profile lowdim-evaluation --asset-root .
```

Use `--asset-root .` for the evaluation commands in the root README. Exact
sizes, SHA-256 digests and source groups are recorded in
[evaluation_assets.json](../example/lowdim/evaluation_assets.json). Data files
are kept byte-for-byte identical to this manifest, including CSV line endings.

## Dataset index

| Dataset | Input point cloud | Independent reference mesh |
|---|---|---|
| ABC_01_00004286 | [CSV](ABC_Official_Selected_8/selected/ABC_01_00004286/ABC_01_00004286_pointcloud_100k.csv) | [OBJ](ABC_Official_Selected_8/selected/ABC_01_00004286/ABC_01_00004286_mesh.obj) |
| ABC_02_00001373 | [CSV](ABC_Official_Selected_8/selected/ABC_02_00001373/ABC_02_00001373_pointcloud_100k.csv) | [OBJ](ABC_Official_Selected_8/selected/ABC_02_00001373/ABC_02_00001373_mesh.obj) |
| ABC_03_00003214 | [CSV](ABC_Official_Selected_8/selected/ABC_03_00003214/ABC_03_00003214_pointcloud_100k.csv) | [OBJ](ABC_Official_Selected_8/selected/ABC_03_00003214/ABC_03_00003214_mesh.obj) |
| ABC_04_00009278 | [CSV](ABC_Official_Selected_8/selected/ABC_04_00009278/ABC_04_00009278_pointcloud_100k.csv) | [OBJ](ABC_Official_Selected_8/selected/ABC_04_00009278/ABC_04_00009278_mesh.obj) |
| ABC_05_00009498 | [CSV](ABC_Official_Selected_8/selected/ABC_05_00009498/ABC_05_00009498_pointcloud_100k.csv) | [OBJ](ABC_Official_Selected_8/selected/ABC_05_00009498/ABC_05_00009498_mesh.obj) |
| ABC_06_00001208 | [CSV](ABC_Official_Selected_8/selected/ABC_06_00001208/ABC_06_00001208_pointcloud_100k.csv) | [OBJ](ABC_Official_Selected_8/selected/ABC_06_00001208/ABC_06_00001208_mesh.obj) |
| ABC_07_00004477 | [CSV](ABC_Official_Selected_8/selected/ABC_07_00004477/ABC_07_00004477_pointcloud_100k.csv) | [OBJ](ABC_Official_Selected_8/selected/ABC_07_00004477/ABC_07_00004477_mesh.obj) |
| ABC_08_00002423 | [CSV](ABC_Official_Selected_8/selected/ABC_08_00002423/ABC_08_00002423_pointcloud_100k.csv) | [OBJ](ABC_Official_Selected_8/selected/ABC_08_00002423/ABC_08_00002423_mesh.obj) |
| ABC_09_00000117 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_09_00000117_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00000117_ce24e0cbec8e45c89735d148_trimesh_009.obj) |
| ABC_10_00000552 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_10_00000552_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00000552_41a24c0b74a34217832f5af0_trimesh_003.obj) |
| ABC_11_00000758 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_11_00000758_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00000758_8ab50540e9804365a8717ec0_trimesh_005.obj) |
| ABC_12_00001063 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_12_00001063_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00001063_c6fda10dc58448d1951c5cc3_trimesh_000.obj) |
| ABC_13_00001430 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_13_00001430_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00001430_a45f6c99e2154f3b94955175_trimesh_014.obj) |
| ABC_14_00001828 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_14_00001828_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00001828_f8ce225b03b045e1a8743a75_trimesh_000.obj) |
| ABC_15_00002199 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_15_00002199_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00002199_6553d5a5a37549379146953a_trimesh_002.obj) |
| ABC_16_00002898 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_16_00002898_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00002898_635bf5c85753420fbc65a4c5_trimesh_000.obj) |
| ABC_17_00003340 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_17_00003340_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00003340_d6d4eff41d95422ca7212c77_trimesh_000.obj) |
| ABC_18_00003838 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_18_00003838_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00003838_72cefc6fb8cf4e0eaa77627b_trimesh_001.obj) |
| ABC_19_00005021 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_19_00005021_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00005021_0a675fbb89444f1a91da0315_trimesh_006.obj) |
| ABC_20_00005613 | [CSV](ABC_Official_Additional_12/pointclouds/ABC_20_00005613_pointcloud_100k.csv) | [OBJ](ABC_Official_Additional_12/meshes/00005613_6d37bbe402a9413ab9629242_trimesh_007.obj) |
| Armadillo | [CSV](Armadillo/armadillo_surface_points_250k_20260526.csv) | [OBJ](source_meshes/Armadillo/armadillo.obj) |
| Bunny | [CSV](Bunny/bunny_surface_points_250k_20260526.csv) | [PLY](source_meshes/Bunny/BunnyMesh.ply) |
| CV2Radial_a020_f16 | [CSV](CV2Radial_a020_f16/cv2_radial_a020_f16.csv) | [NPZ](source_meshes/CV2Radial_a020_f16/CV2Radial_a020_f16_analytic_reference_512x1024.npz) |
| FlowerPetal | [CSV](FlowerPetal/flower_petal_closed_surface_8lobes_250k.csv) | [NPZ](source_meshes/FlowerPetal/FlowerPetal_analytic_reference_512x1024.npz) |
| Knot | [CSV](Knot/knot_surface_points_250k_20260526.csv) | [PLY](source_meshes/Knot/KnotMesh.ply) |
| Monkey | [CSV](Monkey/open3d_monkey_surface_points_250k_20260605.csv) | [OBJ](source_meshes/Monkey/monkey.obj) |
| Sphere | [CSV](Sphere/full_sphere_surface_points_250k_20260528.csv) | [NPZ](source_meshes/Sphere/Sphere_analytic_reference_512x1024.npz) |

## Provenance and use

The source groups are ABC v00 OBJ chunk 0000 (20 datasets), the Stanford 3D
Scanning Repository (Bunny and Armadillo), Open3D MonkeyModel, repository
analytic geometry (Sphere, CV2Radial and FlowerPetal), and author-supplied Knot
geometry. Official source links and applicable use notices are retained in
[DATA_LAYOUT.md](../docs/DATA_LAYOUT.md) and [ARTIFACTS.md](../ARTIFACTS.md).

These files are research inputs. The source-code MIT license does not grant
additional rights in third-party geometry. Preserve the original source
attributions and applicable upstream terms.
