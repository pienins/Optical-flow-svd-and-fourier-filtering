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
| Archived experimental input sets | [`Test_image_sequences/`](Test_image_sequences/) | `.7z` files tracked through Git LFS |
| Archived experimental output sets | [`Example_2WCF-SVD_outputs/`](Example_2WCF-SVD_outputs/) | `.7z` files tracked through Git LFS |

**Note on integration:** The `legacy/2WCF-SVD_v4_4.py` link becomes valid after following [`MERGE_STEPS.md`](MERGE_STEPS.md) and performing the `git mv`. If you have not moved the file yet, the same original file is still at repository root.

## Scientific scope

The current pipeline takes a grayscale TIFF image sequence and estimates 2-D inter-frame displacement using explicit, local-quadratic Farnebäck-type optical flow at multiple pyramid levels. It computes both forward and backward fields, applies robust bidirectional-consistency rejection, and can optionally apply temporal SVD preprocessing, confidence-weighted smoothing, and information-aware regularization. This explicit solver **does not claim numerical identity to OpenCV Farnebäck**. The current code has optional FFT plots of SVD temporal modes; a separate image-frequency Fourier filter has **not** been verified in the current modular pipeline.

**Units:** Flow components are image displacements (pixels/frame pair). For physical velocity, apply known spatial and temporal calibration. Image coordinates are +u right, +v down.

## Quick start: fully local mini benchmark

Requirements: Python 3.10+ with packages in [`requirements-dev.txt`](requirements-dev.txt); optional `wolframclient` is not required for the demonstration.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python benchmarks/generators/generate_mini_uniform.py
python run.py --config examples/mini_uniform_config.json
python benchmarks/evaluators/evaluate_mini_uniform.py
```

The demo generates six synthetic TIFFs (128 × 128, uniform 1 px/frame to the right) and compares five estimated flow pairs against ground truth. Repeating it without clearing the *generated* `output/mini_uniform_run_1` fails intentionally to protect archived results; remove only this disposable demo output or choose a new run ID. See [setup](docs/setup.md).

For experimental processing, copy [`config.json`](config.json) and change `paths.input_images`, `paths.output_root`, `run.runID` and output switches. **The supplied original `config.json` contains a specific Windows `E:/` experimental path and will not run unchanged on other computers.**

## Navigation

| Topic | Documentation |
|---|---|
| Installation, settings, and execution | [Setup](docs/setup.md) |
| Mathematical formulation, mode filtering, consistency, confidence | [Methods](docs/methods.md) |
| Pipeline modules and function map | [Code map](docs/code_map.md) |
| Benchmark generation and validation metrics | [Benchmarks](docs/benchmarks.md) |
| Actual figure examples and portable mini demo | [Examples](docs/examples.md) |
| Data types and output schemas | [Output formats](docs/output_formats.md) |
| Current features vs unverified capabilities | [Current state](docs/current_state.md) |
| Known issues and scientific caveats | [Known limitations](docs/known_limitations.md) |
| Existing experimental history | [Legacy material](docs/legacy_material.md) |
| Historical run indexing and LFS | [Result registry](docs/results_registry.md) |
| Missing inputs/codes before publication | [Missing materials](docs/missing_materials.md) |
| Integrate into your GitHub `main` branch | [Step-by-step merge](MERGE_STEPS.md) |

## Results and reproducibility

Existing source-repository historical `.7z` archives are **left in their original locations** (Git LFS). The additional [`results/examples/`](results/examples/) tree contains **selected excerpts** of two later uploaded 99-frame-pair output archives (uniform translation and Rankine vortex), plus a tiny newly generated uniform example. Neither later archive contains the corresponding original TIFF inputs or independent truth-map files, so the archived internal consistency/retention statistics are *not* independent ground-truth accuracy measurements. See [Missing materials](docs/missing_materials.md) and [Result registry](docs/results_registry.md).

**Do not delete archived runs, compare unmatched configurations as a controlled experiment, or publish accuracy claims based only on the saved QC logs.** The uploaded historical Rankine configuration used `frame2_polynomial` warping, which is not exposed by the current modular solver.
