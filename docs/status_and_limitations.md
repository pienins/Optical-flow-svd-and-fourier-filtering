# Status and limitations

> **Warning — measured accuracy problem (§1).** With the shipped `config.json` (`InformationRidgeFraction = 2`), the solver underestimated displacement by about 20 % on a uniform-translation benchmark, while every vector passed the consistency check. Results produced with this setting may be biased low. `InformationRidgeFraction` ≤ 0.1, or disabling the information-aware solver, gave errors below 0.4 % on the same test.

This document records what is implemented, what has been shown to be wrong, what could mislead, and what is still missing. §1 lists errors measured against ground truth. The other sections list hazards and gaps; they are not by themselves evidence that a particular result is incorrect.

Last reviewed: 24 November 2026, against the `modular-2wcf` branch.

## 1. Measured problems

**Information ridge underestimates displacement.** With the current `config.json` (`InformationRidgeFraction = 2`, 2 iterations per level), a uniform-translation benchmark (512×512, 3 frame pairs, seed 42) recovered only about 80 % of the true displacement, while 100 % of vectors passed the consistency check:

| Truth (u, v) [px/frame] | Ridge fraction | Spatial prior | Median EPE [px] | Relative EPE |
|---|---|---|---|---|
| (1, 0) | 2 | on | 0.197 | 19.7 % |
| (1, 0) | solver off | – | 0.0076 | 0.76 % |
| (2.5, −1.5) | 2 | on | 0.543 | 18.6 % |
| (2.5, −1.5) | 2 | off | 0.543 | 18.6 % |
| (2.5, −1.5) | 0.5 | on | 0.041 | 1.4 % |
| (2.5, −1.5) | 0.1 | on | 0.010 | 0.35 % |
| (2.5, −1.5) | 0.01 | on | 0.010 | 0.36 % |
| (2.5, −1.5) | solver off | – | 0.010 | 0.36 % |

The bias comes from the ridge fraction, not the spatial prior or the increment cap, and matches the damping described in `methods.md` §4.3. These are three frame pairs of one easy synthetic flow; they do not show whether a smaller ridge is optimal for experimental data, but they show that results obtained with fraction 2 can be biased low by about 20 % without any sign in the consistency statistics.

## 2. Numerical and implementation hazards

- **Non-finite flow stalls the pipeline instead of failing.** If the solver ever returned NaN or infinite displacements, the consistency step (`cv2.remap` sampling the backward flow at `x + d12`, reflected borders) would not return. With NaN coordinates on a 48×48 field, a single `cv2.remap` call was still running after 15 s, so the run would hang with no error. With finite input images the solver is not expected to produce non-finite flow, so this is a latent risk, not a known failure. Planned guard: check `d12` and `d21` for non-finite values before resampling and raise an error. For the same reason, the statistics fallback in `methods.md` §5.2 cannot be reached through a normal run.
- **Increment cap limits the recoverable displacement.** `ExplicitMaxResidualIncrement` bounds every iteration, so starting from zero flow the solver cannot recover more than about $N\,\delta_{\max}\sum_\ell s^{-\ell}$ (`methods.md` §4.1): about 19 px for a 512-px image with the current settings, less for smaller images. A benchmark near this bound tests the cap, not the estimator.
- **Parameter values are not interchangeable with OpenCV.** `PolyFitKernelSize` is a full width (OpenCV's `poly_n` is a half-width and only accepts 5 or 7), and `PyramidLevels` counts one level more than OpenCV's `levels`. Results from OpenCV runs and from this solver must be compared with explicitly mapped parameters (`methods.md` §7).
- **Border sensitivity.** The backward `cv2.remap` uses reflected borders, so consistency errors near image margins and for inflow/outflow are not meaningful. Report both all-grid and interior-region statistics.
- **Memory.** The full input stack, and with SVD the $T\times HW$ matrix, stay in memory for the whole run. The bounded worker queue limits dense-flow memory only (`code_map.md`).
- **Rankine evaluator not migrated.** `benchmarks/evaluators/evaluate_2wcf_rankine_test.py` still contains local `E:/...` paths (edit them before running) and its own copies of the metric functions.
- **Rankine centre estimator is biased for finite displacement.** The evaluator's centre estimate assumes $\mathbf d\cdot(\mathbf x-\mathbf c)=0$, which holds for an infinitesimal tangent vector. For a finite rotation chord the correct relation is $\mathbf d\cdot(\mathbf x-\mathbf c)=-\|\mathbf d\|^2/2$. Correct and validate this before citing centre error (`benchmarks.md`). The binned core radius is a proxy only.

## 3. Interpretation hazards

- **Consistency is not accuracy.** A pair of mutually consistent but wrong fields passes the two-way check; §1 is a measured example. Very high retention does not imply low endpoint error.
- **Selection bias.** Retained-only median EPE can improve because the method rejects the hardest vectors. Report error on the common grid, retained fraction, and ideally comparisons at matched retention.
- **Displacement, not velocity.** Outputs are in px per frame pair. SI velocity requires known pixel size and frame interval.

## 4. Historical runs

These apply to the archived `synthetic_uniform_u8_vm20_run_55` and `synthetic_rankine_R80_vb3_cx255p5_cy255p5_run_11` outputs (`history/examples.md`, `history/results_registry.md`).

- **Inputs and ground truth missing.** The output archives contain fields and diagnostics, but not the input TIFF sequences or the `true_vector_values.npz` that generated them. Their EPE and angular error cannot be independently recalculated, and the runs cannot be repeated exactly.
- **Rankine run used a solver mode that no longer exists.** Its `parameters.json` specifies `FarnebackWarpMode = "frame2_polynomial"` with `Frame2PolynomialRefinementIterations`, `Frame2PolynomialNewtonSteps` and related options. The current solver accepts only `image` and `coefficients`. Its outputs must not be attributed to the current solver, and reproducing it needs the historical source version.
- **Uniform archive lacks warp-mode provenance.** Its `parameters.json` does not record `FarnebackWarpMode`.
- **Different parameter defaults.** The current `config.json` uses `PyramidDownsampleFactor = 0.45`, `PyramidLevels = 7`, `WindowSize = 39`, `IterationCount = 2`, `PolyFitKernelSize = 5`, `GaussKernelStdev = 2.5`, `InformationRidgeFraction = 2`, `RelativeConsistencyAlpha = 0.2`. Both archived runs report mostly `0.5`, `8`, `28`, `5`, `7`, `1`, `0.5`, `0.01`. Do not compare them as though they differed by one feature.
- **Keep historical data immutable.** Store complete archives unchanged and reference them by checksum. Do not replace 99-pair results with a few plots or overwrite archives; make corrections in new, derived directories.

## 5. Missing materials

Unknown here means "not among the inspected files", not "does not exist".

- **Rankine generator** actually used for the `R80_vb3` run. The evaluator documents the expected ground-truth arrays but is not a generator. A newly written generator must be labelled as a separate experiment.
- **Run provenance.** `run_manifest.json` does not record the Git commit, package versions, input checksums or random seeds. Record exact source commit, dependency versions, platform, input hashes, seeds and timestamp for every run used in a publication.
- **The two 99-pair result archives** (about 84 and 181 MiB). `results/archive_checksums.json` records their checksums, but the archives are not yet in `results/archives/`.
- **Other historical results**, including full parameter sweeps, PSO optimization/training/validation splits and independent-seed tests. The LFS archives in `experimental_data/` are not the complete historical record.
- **Benchmark comparison protocol**: shared evaluation masks and populations, radial/interior masks, zero-motion handling, error–retention reporting and fixed baseline parameters.
- **Licensing, credit and citation** (`LICENSE`, `CITATION.cff`, authors, attributions). These cannot be generated reliably from code.
- **Data availability and Git LFS plan.** Decide where large data lives long term, check the Git LFS quota (download bandwidth is charged to the repository owner), and keep checksums so collaborators can verify full files, not only LFS pointers.
- **Testing on real data and an environment lock.** The tests (`tests/`) are smoke and regression checks on small synthetic data; they do not establish robustness on full-resolution experimental sequences or across platforms.
- **Ground-truth figures**: estimated vs true vector fields, spatial EPE and retention maps, radial EPE and tangential profiles for Rankine, and matched ablation comparisons. Archived QC plots alone do not demonstrate accuracy.

## 6. Implementation status

| Component | Status | Evidence / notes |
|---|---|---|
| Explicit local-quadratic Farnebäck solver | Implemented; warp modes `image` and `coefficients` | `two_wcf/solver.py`; `tests/test_smoke.py` |
| Temporal SVD mode removal | Implemented, optional | `two_wcf/image_io.py`; smoke test with SVD |
| Fourier-domain image filtering | **Not implemented** | FFT is used only to analyze SVD modes, in `Artifact_mode_detection/` (`methods.md` §1.1) |
| Two-way consistency with adaptive MAD threshold | Implemented | `two_wcf/consistency.py` |
| Confidence-weighted smoothing, local sigma | Implemented, optional | `two_wcf/consistency.py`; smoke test |
| Information ridge and spatial prior | Implemented, optional | `two_wcf/solver.py`; **biased low with fraction 2 (§1)** |
| Run manifest and overwrite protection | Implemented | `two_wcf/pipeline.py`, `two_wcf/outputs.py` |
| Uniform-flow benchmark (generate, evaluate) | Implemented, config-driven | `benchmarks/homogeneous_constant_flow/`; end-to-end test in `tests/test_benchmarks.py` |
| Rankine benchmark | Evaluator only; generator missing | `benchmarks/evaluators/` |
| Motionless, shear-flow and Gaussian-blob benchmarks | Not yet implemented | scripts not yet in the repository |
| Historical uniform and Rankine 99-pair outputs | Excerpts only | `results/examples/`; full archives not yet added (§5) |
| Inputs and ground truth of the archived runs | Not in the repository | §4 |
| Parameter-sweep / PSO archive | Not in the repository | §5 |
