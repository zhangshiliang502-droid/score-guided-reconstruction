# Supported and validation environments

## Production target

| Component | Requirement |
|---|---|
| Host | 64-bit Linux compute server |
| Python | 3.10--3.12 |
| Accelerator | NVIDIA CUDA-capable GPU for full experiments |
| PyTorch | `>=2.1,<3.0`, built for a CUDA runtime supported by the driver |
| Scientific stack | Versions constrained by `requirements.txt` or `environment.yml` |

Confirm accelerator visibility before a production run:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The NVIDIA driver, CUDA runtime and PyTorch build must be mutually compatible.

## CI validation matrix

The small-noise experiment's reporting dependencies, pandas and Matplotlib,
are specified in
`experiments/empirical_vs_trained_small_sigma/requirements.txt`.

The included GitHub Actions workflow installs the method and reporting
dependencies and runs source validation and unit tests on
GitHub-hosted Ubuntu runners with Python 3.10, 3.11 and 3.12. Its Python 3.11
job also runs the reduced end-to-end CPU training test. Routine CI does not
download Git LFS results and does not certify production CUDA performance.

Run the same software checks inside the intended server allocation:

```bash
python tools/validate_release.py --profile source
python -m unittest discover -s tests -v
python tools/smoke_test_lowdim.py
python tools/reproduce_highdim.py --examples all --dry-run
```

Successful execution of the workflow establishes hosted-Linux source
validation for the corresponding commit. It does not establish production-GPU
benchmark certification; numerical results and their recorded configurations
are distributed separately from CI evidence.
