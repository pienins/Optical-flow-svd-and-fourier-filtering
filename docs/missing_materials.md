# Missing materials and publication blockers

Inventory based on the original `main` source ZIP, modular explicit solver ZIP and two separately uploaded later result archives (17 Nov 2026). **Unknown does not mean nonexistent; it means not among the inspected files.**

## Needed to reproduce the two additional full benchmark runs

1. **Exact synthetic input TIFF sequences** for `synthetic_uniform_u8_vm20_run_55` and `synthetic_rankine_R80_vb3_cx255p5_cy255p5_run_11`. The uploaded output `.7z` archives contain fields and diagnostics, not their corresponding original input frames.
2. **Independent ground-truth `true_vector_values.npz` files** matching those exact sequences, plus generator `parameters.json`, seed and numerical conventions. Without this, EPE and angular-error claims for those saved outputs cannot be independently validated.
3. **Rankine generator source** actually used for the `R80_vb3` run. The available Rankine evaluator describes its expected truth fields, but is not a generator. Do not substitute a newly written generator without labeling it as a separate experiment.
4. **Original algorithm source version for the Rankine archived run** using `FarnebackWarpMode="frame2_polynomial"`. Current modular refactor permits `"image"` and `"coefficients"`; these are not automatically interchangeable with the historical mode.
5. **Run provenance:** complete parameter snapshots, exact code commit or SHA-256, dependency versions, platform, input file hash list, random seeds, timestamp and hardware for original performance results.

## Needed for a publication-grade archive beyond these two examples

6. **All other historical results and raw run metadata**, including full parameter sweeps, PSO optimization/training/validation splits and independent-seed tests, where applicable. `main` provides several LFS archives; they do not establish that the complete historical record has been uploaded.
7. **Precise benchmark comparison protocol:** shared evaluation masks/populations, radial/interior masks, zero-motion handling, error/retention trade-off reporting and fixed baseline parameters.
8. **Corrected or validated Rankine centre estimation.** Existing validator's dot-product center equation is exact for infinitesimal tangent vectors, but not finite rotation chords; confirm the truth convention and correct before reporting centre error. See [`benchmarks.md`](benchmarks.md).
9. **Author-approved licensing/credit and citations** (`LICENSE`, `CITATION.cff`, responsible authors, source attributions); these cannot be generated reliably from code alone.
10. **Permanent public data availability and Git LFS quota plan** for large images/results, including checksums and whether collaborators will be able to retrieve the full data, not only LFS pointer files.
11. **Tests on real full-resolution experimental data and OS/environment lock:** supplied smoke tests and one small demo do not establish numerical equivalence or broad robustness.
12. **Figures for actual ground-truth comparisons**: sampled estimated vs GT vector fields, spatial EPE/retention maps, radial EPE/tangential profiles for Rankine, and matched ablation comparisons. Available archived QC visuals alone do not demonstrate accuracy.

## Incomplete evaluator integration

The supplied original validators at `benchmarks/evaluators/evaluate_2wcf_uniform_test.py` and `evaluate_2wcf_rankine_test.py` include absolute Windows `E:/` paths. They must be modified/configured for each dataset before running. Their validation NPZ schema matches the current modular pipeline, but results cannot be regenerated without the exact ground-truth arrays and inputs.

The original source-repository uniform generator has a misleading final console message (`ground_truth.npz` vs actual `true_vector_values.npz`); preserve that historical version and correct this in a **separate code-change commit** if desired.
