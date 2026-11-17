# Known limitations and validation hazards

These notes identify reproducibility/scientific interpretation issues; they are **not** evidence that a particular numeric result is incorrect.

1. **Missing raw material.** Both uploaded `.7z` files contain solver outputs but not their actual input TIFFs or independent truth `.npz`. Exact historical reruns are unavailable from the uploads alone.
2. **Historical solver mismatch.** The Rankine `parameters.json` specifies `FarnebackWarpMode="frame2_polynomial"`, `Frame2PolynomialRefinementIterations`, `Frame2PolynomialNewtonSteps` and related options absent from `two_wcf/config.py`. The supplied modular implementation accepts only `image` or `coefficients`. The historical outputs must not be attributed to an identical current solver.
3. **Uniform archive lacks warp-mode provenance.** The archived uniform `parameters.json` does not record `FarnebackWarpMode`; the active mode cannot be inferred with certainty.
4. **Different parameter defaults.** The current `config.json` uses `PyramidDownsampleFactor=0.45`, `PyramidLevels=7`, `WindowSize=39`, `IterationCount=2`, `PolyFitKernelSize=5`, `GaussKernelStdev=2.5`, `InformationRidgeFraction=2`, `RelativeConsistencyAlpha=0.2`. Both archived runs report mostly `0.5`, `8`, `28`, `5`, `7`, `1`, `0.5`, `0.01`. Do not compare runs as though they differed by one feature only.
5. **No independent accuracy from forward/backward residual.** A pair of mutually consistent but wrong fields can pass 2WCF. Very high retention does not imply very low EPE.
6. **Selection bias.** Retained-only median EPE can improve when the method rejects the hardest vectors. Report common-grid raw error, retained fraction, and ideally matched-retention comparisons.
7. **Finite Rankine displacement.** In `evaluate_2wcf_rankine_test.py`, the center-estimation equation assumes tangent orthogonality. For a finite rotation chord it is biased: correct circle geometry satisfies `d·(x−c) = −|d|²/2`. The binned core radius is a proxy only.
8. **Velocity terminology.** Code outputs displacement `px/frame pair`; SI velocity requires known spatial/time calibration.
9. **Input/global intensity and memory.** Current code loads the full stack, and SVD can be costly for large images/time counts. Bounds on concurrent pair tasks do not cap input-stack memory.
10. **Hard-coded validator/generator paths.** The three historical scripts in `benchmarks/` contain local `E:/...` paths. Update them before running. The uniform generator prints `ground_truth.npz` but saves `true_vector_values.npz`.
11. **Incomplete source metadata.** The current manifest does not include a Git commit hash, Python package version lockfile, image file checksums or random seed (unless recorded elsewhere). These should be added for final publication reproducibility.
12. **Optional Fourier transform.** The FFT call in SVD diagnostics is not a Fourier-domain filter; document it accordingly.
13. **Potential sensitivity to borders / sampling.** The default backward `cv2.remap` uses reflected borders. Error metrics may be optimistic or pessimistic near margins, so report both all-grid and interior-region statistics.
14. **Historical results volume.** Complete data should be stored immutably and referenced by checksums. Do not replace 99-pair results with only a few plots or silently overwrite archives.
