# Optical flow with SVD and Fourier-mode analysis — 2WCF research repository

**Status:** research software in development; latest supplied modular explicit 2WCF implementation plus archived historical scripts and results. This repository is intended as a reproducible companion to ongoing scientific work, not a validated production tool.

## Which implementation should I use?

| Component | Location | Status |
|---|---|---|
| **Modular 2WCF-SVD implementation** | [`run.py`](run.py), [`two_wcf/`](two_wcf/), [`config.json`](config.json) | **Current development version** from the supplied 17 Nov 2026 refactor |
| Historical monolithic implementation | [`legacy/2WCF-SVD_v4_4.py`](legacy/2WCF-SVD_v4_4.py) | **Historical**; preserved to reproduce prior work |
| Artifact/SVD mode analysis | [`Artifact_mode_detection/`](Artifact_mode_detection/) | Historical Python/Mathematica tools and examples |
| Mathematica field postprocessing | [`OFV_field_postprocessing_v2.nb`](OFV_field_postprocessing_v2.nb) | Historical notebook |
| Original uniform-flow utilities | [`Generation_testing_unfirom_flow/`](Generation_testing_unfirom_flow/) | Historical, retained without edits |
| Experimental image sequences and their outputs | [`experimental_data/`](experimental_data/) | `.7z` files tracked through Git LFS; no ground truth |

## Scientific scope

The current pipeline takes a grayscale TIFF image sequence and estimates 2-D inter-frame displacement using explicit, local-quadratic Farnebäck-type optical flow at multiple pyramid levels. It computes both forward and backward fields, applies robust bidirectional-consistency rejection, and can optionally apply temporal SVD preprocessing, confidence-weighted smoothing, and information-aware regularization. This explicit solver **does not claim numerical identity to OpenCV Farnebäck**. The current code has optional FFT plots of SVD temporal modes.

**Units:** Flow components are image displacements (pixels/frame pair). For physical velocity, apply known spatial and temporal calibration. Image coordinates are +u right, +v down.

## Quick start

Requirements: Python 3.10+ with packages in [`requirements-dev.txt`](requirements-dev.txt); `wolframclient` is needed only when `outputs.WriteWXF=true`.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python run.py --config /path/to/your_config.json
```

To process a sequence, copy [`config.json`](config.json) and change `paths.input_images`, `paths.output_root`, `run.runID` and output switches. **The supplied `config.json` contains a specific Windows `E:/` experimental path and will not run unchanged on other computers.** See [setup](docs/setup.md).

Synthetic benchmarks with known ground truth are in [`benchmarks/`](benchmarks/) and are being reorganized by flow scenario; see [Benchmarks](docs/benchmarks.md).

## Navigation

| Topic | Documentation |
|---|---|
| Installation, settings, and execution | [Setup](docs/setup.md) |
| Mathematical formulation, mode filtering, consistency, confidence | [Methods](docs/methods.md) |
| Pipeline modules and function map | [Code map](docs/code_map.md) |
| Benchmark generation and validation metrics | [Benchmarks](docs/benchmarks.md) |
| Data types and output schemas | [Output formats](docs/output_formats.md) |
| Implementation status, known problems, missing materials | [Status and limitations](docs/status_and_limitations.md) |
| Per-experiment release checklist | [Reproducibility checklist](docs/reproducibility_checklist.md) |
| Historical example figures | [History: examples](docs/history/examples.md) |
| Historical material in this repository | [History: legacy material](docs/history/legacy_material.md) |
| Historical run registry, checksums and LFS | [History: results registry](docs/history/results_registry.md) |

## Results and reproducibility

Existing source-repository historical `.7z` archives are **left in their original locations** (Git LFS). The additional [`results/examples/`](results/examples/) tree contains **selected excerpts** of two later uploaded 99-frame-pair output archives (uniform translation and Rankine vortex). Neither later archive contains the corresponding original TIFF inputs or independent truth-map files, so the archived internal consistency/retention statistics are *not* independent ground-truth accuracy measurements. See [Status and limitations](docs/status_and_limitations.md) and the [results registry](docs/history/results_registry.md).

**Do not delete archived runs, compare unmatched configurations as a controlled experiment, or publish accuracy claims based only on the saved QC logs.** The uploaded historical Rankine configuration used `frame2_polynomial` warping, which is not exposed by the current modular solver.
