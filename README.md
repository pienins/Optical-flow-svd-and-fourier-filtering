# Optical flow with SVD and Fourier-mode analysis — 2WCF research repository

**Status:** research software in development: the current modular explicit 2WCF implementation, plus archived historical scripts and results. This repository is intended as a reproducible companion to ongoing scientific work, not a validated production tool.

## Which implementation should I use?

| Component | Location | Status |
|---|---|---|
| **Modular 2WCF-SVD implementation** | [`run.py`](run.py), [`two_wcf/`](two_wcf/), [`config.json`](config.json) | **Current development version** |
| Historical monolithic implementation | [`legacy/2WCF-SVD_v4_4.py`](legacy/2WCF-SVD_v4_4.py) | **Historical**; preserved to reproduce prior work |
| Artifact/SVD mode analysis | [`Artifact_mode_detection/`](Artifact_mode_detection/) | In use: Python/Mathematica tools for choosing `KillIDs`, with examples. The Python script needs `scikit-image` and `wolframclient` |
| Mathematica field postprocessing | [`OFV_field_postprocessing_v2.nb`](OFV_field_postprocessing_v2.nb) | Historical notebook |
| Original uniform-flow utilities | [`Generation_testing_unfirom_flow/`](Generation_testing_unfirom_flow/) | Historical, retained without edits |
| Experimental image sequences and their outputs | [`experimental_data/`](experimental_data/) | `.7z` files tracked through Git LFS; no ground truth |

## Scientific scope

The current pipeline takes a grayscale TIFF image sequence and estimates 2-D inter-frame displacement using explicit, local-quadratic Farnebäck-type optical flow at multiple pyramid levels. It computes both forward and backward fields, applies robust bidirectional-consistency rejection, and can optionally apply temporal SVD preprocessing, confidence-weighted smoothing, and information-aware regularization. This explicit solver **does not claim numerical identity to OpenCV Farnebäck**. Frequency analysis of the SVD temporal modes is done separately, in `Artifact_mode_detection/`.

**Units:** Flow components are image displacements (pixels/frame pair). For physical velocity, apply known spatial and temporal calibration. Image coordinates are +u right, +v down.

## Quick start

On Windows with Anaconda (details and other systems in [setup](docs/setup.md)):

1. Clone with GitHub Desktop.
2. Once, in Anaconda Prompt in the repository folder: `flowenv_setup.bat`
3. Each session: `conda activate flowenv`, then `spyder`
4. Set `paths.input_images`, `paths.output_root` and a new `run.runID` in [`config.json`](config.json), then run `run.py`.

**`config.json` contains a specific Windows `E:/` experimental path and will not run unchanged on other computers.**

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
