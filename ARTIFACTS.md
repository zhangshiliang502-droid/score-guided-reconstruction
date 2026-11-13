# Research artifacts and third-party sources

The MIT License in this repository applies to repository-authored source code,
documentation, configurations and analytic definitions. It does not grant
rights in third-party source geometry or automatically relicense numerical
artifacts derived from that geometry.

## Retained binary results

The selected checkpoints and reconstructed meshes under `example/lowdim/`,
and the 27 input clouds and 27 reference meshes under `datasets/`, are
provided as research artifacts for inspection and reproducibility. Their
presence in this repository does not create a separate license grant for any
third-party content represented by or used to produce them. Users are
responsible for complying with the terms of the corresponding source data.

| Result group | Source and applicable terms |
|---|---|
| ABC examples | [ABC Dataset](https://deep-geometry.github.io/abc-dataset/); copyright in the source CAD models remains with their creators, subject to the terms identified by the dataset provider |
| Bunny and Armadillo | [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/); follow its attribution, research-use and redistribution conditions |
| Monkey | [Open3D MonkeyModel](https://www.open3d.org/docs/release/python_api/open3d.data.MonkeyModel.html); confirm and follow the terms supplied with the upstream asset |
| Knot | Author-supplied source geometry; no additional rights are granted by this repository beyond rights held by the contributor |
| Sphere, CV2Radial and FlowerPetal | Generated from repository-authored analytic definitions; the definitions themselves are covered by the MIT License |

The manifest-identified reference geometry and author-generated training point
clouds are included through Git LFS. Their exact file identities are recorded in
`example/lowdim/evaluation_assets.json`; download information is provided in
[`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

## Evaluation records

Tables and metric records document the reported experiments. Citations and
attributions to the source datasets remain required where applicable. This
notice supplements, and does not replace, the terms published by each source.
