# Current implementation and research status

Status documented from uploads dated 17 November 2026; **not** a universal claim about the project's newest unpublished work.

| Component | Implementation | Evidence in supplied files |
|---|---|---|
| Explicit local-quadratic Farnebäck solver | Yes; two warp modes (`image`, `coefficients`) | Modules and smoke tests |
| Optional temporal SVD mode removal | Yes | `image_io.py`; smoke test with SVD |
| Fourier filtering as an independent algorithmic step | **No evidence** | FFT appears for temporal-mode visualization only |
| Bidirectional consistency + adaptive MAD threshold | Yes | `consistency.py`, 99-pair QC logs |
| Confidence-weighted smoothing and local sigma | Yes, optional | `consistency.py` and smoke test |
| Information-based ridge + spatial flow prior | Yes, optional | `solver.py`, diagnostic maps |
| Output manifest and collision protection | Yes in refactor | `pipeline.py`, `outputs.py`, smoke tests |
| Historical uniform 99-pair run outputs | Available as original `.7z` | QC CSV, diagnostics, validation NPZ |
| Historical Rankine 99-pair outputs | Available as original `.7z` | Same, with unsupported historical warp mode |
| Original input TIFFs for the two archived runs | **Not supplied** | Cannot reproduce full results exactly |
| Independent ground-truth NPZ for the two archived runs | **Not supplied** | EPE and angular accuracy cannot be independently recalculated |
| Rankine synthetic image generator | **Not supplied** | Formula-level description only |
| Full parameter-sweep / PSO archive | **Not supplied in this upload** | Add as later archival tranche |

## What is tested

The current ZIP includes seven local smoke tests covering configuration checks, two warp modes, information solver toggles, streaming output and SVD/local-confidence mode. Seven tests passed during preparation of this package. This does not verify that the historical archived runs were produced by this exact code version. A fair comparison still needs identical inputs, seeds and params.

## Next repository milestones

1. Import the *complete* historical archives through Git LFS, along with original TIFFs and exact truth maps.
2. Recover the historical solver implementation used for `frame2_polynomial` or explicitly mark that run as not directly reproducible.
3. Attach per-run software commit hash, package versions, data hashes and generator settings.
4. Recompute independent EPE and angular errors using the correct ground-truth convention, including a finite-displacement Rankine ground truth.
5. Perform controlled ablations at fixed input seeds and report error–retention trade-offs.

## Historical `main` repository

The inspected main-branch ZIP already contains v4.4, SVD/Fourier auxiliary analysis, post-processing Mathematica notebook, four LFS-tracked image archives and six LFS-tracked output archives. The integration overlay does not alter their data content; see [legacy material](legacy_material.md).
