# Additional real-shape experiment inputs

The retained real-shape experiment uses the exact file identities listed in
[`protocol.json`](../results/real_shapes/protocol.json). Bunny's input cloud
and reference mesh, and Flower's input cloud, are included under the root
`datasets/` directory. The following additional inputs are not bundled.

## Gargoyle oriented scan

The experiment uses the oriented Gargoyle scan distributed by the
NeuralPull-Pytorch project. From the repository root, download it under the
name expected by the launcher:

```bash
curl -L \
  https://raw.githubusercontent.com/mabaorui/NeuralPull-Pytorch/master/data/gargoyle.ply \
  -o experiments/empirical_vs_trained_small_sigma/assets/gargoyle_with_normals.ply
```

Expected SHA-256:

```text
78bb925be84a1eced067eb21f4020417731b42c6256b21e86711ce54ee2c9b67
```

Set `GARGOYLE_SURFACE=/absolute/path/to/gargoyle.ply` to use a copy stored
elsewhere.

## Gargoyle normalization point cloud

The retained experiment also uses a separate 100,000-point CSV to define its
mean/max-radius normalization. Supply it as `GARGOYLE_POINT_CLOUD`; downloading
the oriented scan alone does not supply this exact sample.

Expected SHA-256:

```text
cf7881065823ec0ec16ce17c46b98887d15eeb10755df43792a95a2bf5394efa
```

This repository does not provide an acquisition URL for that exact derivative
sample. Without it, the Gargoyle results can be inspected, but the retained
run cannot be replayed from bundled inputs alone. Use `DATASETS=bunny` to run
the comparison with inputs included in the repository.

## Flower reference mesh

Set `FLOWER_SURFACE` to the NPZ reference mesh used for the retained experiment.
Its expected SHA-256 is:

```text
274b9b53b08f3b772a555fe7138064dc1dff4c85d636012e4303f5cf9553f83c
```

The bundled Flower reference mesh has a different file digest. It can be
supplied through `FLOWER_SURFACE` for a new run, but that changes the recorded
input identity. This repository does not provide a separate download URL for
the exact reference file used by the retained real-shape experiment.

Source attribution and applicable upstream terms remain required for all
external geometry.
