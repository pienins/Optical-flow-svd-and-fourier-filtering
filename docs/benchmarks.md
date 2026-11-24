# Benchmark generation and validation

## Ground-truth conventions

Optical flow is a finite **inter-frame displacement**, not necessarily the instantaneous Eulerian velocity at the first frame. In a synthetic test, ground truth should use the same temporal correspondence and sampling locations as the estimator. With image coordinates $x$ right and $y$ down, the measured displacement is $\mathbf d=(u,v)$ in `px/frame pair` (typically written `px/frame` for consecutive frames). To convert to `mm/s`, multiply by `pixel_size_mm / frame_interval_s`.

## Layout

Benchmarks are organized by flow scenario. Each scenario folder holds its source and configuration; generated images and evaluation results are written to the locations set in its `config.json` (relative paths are resolved against that file).

```text
benchmarks/
├── common/metrics.py               shared metric definitions (EPE, angular error, safe statistics)
├── homogeneous_constant_flow/      generate.py, evaluate.py, config.json
└── evaluators/                     Rankine evaluator, not yet migrated (generator missing)
```

`motionless/`, `shear_flow/`, `rankine_vortex/` and `gaussian_blob/` will be added when their scripts are available.

## Homogeneous constant flow (`benchmarks/homogeneous_constant_flow/`)

```bash
python benchmarks/homogeneous_constant_flow/generate.py        # writes frames + ground truth
python run.py --config /path/to/pipeline_config.json           # with diagnostics.SaveValidationData = true
python benchmarks/homogeneous_constant_flow/evaluate.py        # after setting paths.pipeline_output_dir
```

Both scripts read the same `config.json` (or `--config PATH`). Its default settings are 512×512 images, 100 frames, ~4000 visible tracer particles, Gaussian particle width 1.1 px, approximate peak 220, random seed 42 and `(u, v) = (1, 0)` px/frame. Do **not** mistake this for the historical `(8, −20)` run. Every particle center evolves as

$$x_i(t)=x_i(0)+t u_0,\qquad y_i(t)=y_i(0)+t v_0.$$

Subpixel positions are bilinearly splatted onto four pixels, Gaussian blurred, fixed-intensity scaled, clipped to 8-bit, and saved as TIFF. Particles are initially seeded beyond the image extent to support entries/exits. Output goes to `<generated_root>/synthetic_uniform_u<u>_v<v>/`, with decimal points written as `p` and minus signs as `m` (e.g. `synthetic_uniform_u8_vm20`, matching the historical archive name). That folder contains `frames/`, `true_vector_values.npz` (arrays `u_gt`, `v_gt`; scalars `true_u`, `true_v`, `true_speed`) and `parameters.json`. The generator refuses to write into a folder that already contains frames unless `generate.overwrite_existing = true`, so frames from an earlier, longer sequence cannot be silently mixed in.

The evaluator derives the ground-truth folder from the same flow parameters, so only `paths.pipeline_output_dir` must be set to the run directory produced by `run.py`. It writes `uniform_validation_per_frame.csv`, `uniform_validation_summary.txt` and four plots to `<pipeline_output_dir>/synthetic_validation_results/`. It refuses runs recorded with `VelocityInversion ≠ 1`, because their `u_output`/`v_output` are no longer in image coordinates.

These scripts were migrated from the originals (still preserved in `Generation_testing_unfirom_flow/`). With identical settings, the migrated generator produced byte-identical frames, ground truth and `parameters.json`, and the migrated evaluator produced identical CSV, summary and plots on the same pipeline run (checked 24 Nov 2026). `tests/test_benchmarks.py` runs generate → pipeline → evaluate on a small sequence.

## Rankine vortex benchmark

A Rankine vortex has piecewise tangential speed

$$
v_{\theta}(r)=\begin{cases}\Omega r,& r\le R,\\ \Omega R^2/r,&r>R,\end{cases}
\qquad \Omega=V_R/R,
$$

with $r=\sqrt{(x-c_x)^2+(y-c_y)^2}$. A tracer orbit has angular increment $\Delta\theta=\Omega\Delta t$ inside the core and $\Delta\theta=\Omega R^2\Delta t/r^2$ outside. For an exact finite-displacement benchmark, rotate the initial point by $\Delta\theta$, then define **ground truth as the final minus initial Cartesian position** (the chord), not the local tangent multiplied by $\Delta t$. This is particularly important at large angular displacements. These equations describe a suitable Rankine benchmark; **the Rankine generator used for the archived run is not in this repository**, so its implementation-specific details are not yet verified.

The Rankine evaluator expects `true_vector_values.npz` keys `u_gt`, `v_gt`, `speed_gt`, `radius_gt`, `radial_gt`, `tangential_gt`, `continuous_tangential_speed_gt`, `continuous_vorticity_gt`, `continuous_divergence_gt`, `center_x`, `center_y`, `core_radius_px`, `boundary_tangential_speed_px`, `omega_core_rad_per_frame`. The historical run's name indicates `R80`, `vb3`, and `(cx,cy)=(255.5,255.5)`; **filenames alone are not adequate proof of the generating parameter values**.

## Evaluators

Both evaluators read `validation_*.npz`, look up ground truth at the sampled `x`, `y`, and report per-pair and summary metrics, with a 24-pixel interior exclusion by default. The Rankine evaluator (`benchmarks/evaluators/evaluate_2wcf_rankine_test.py`) additionally uses a minimum true speed of 0.25 px/frame for angular/relative error, 14 radial bins, radial/tangential bias, center error and a binned core-radius proxy. It has not been migrated yet: **edit `SYNTHETIC_TEST_DIR` and `PIPELINE_OUTPUT_DIR`** in the script before use, and it still contains its own copies of the metric functions.

### Metrics

Endpoint error (EPE) compares estimated and independently known vectors at the same sample point:

$$\mathrm{EPE}_i=\sqrt{(u_i-\hat u_i)^2+(v_i-\hat v_i)^2}.$$

The two vectors' angular error is $\theta_i=\cos^{-1}[(\mathbf d_i\cdot\hat{\mathbf d}_i)/(\|\mathbf d_i\|\|\hat{\mathbf d}_i\|)]$; it is undefined if either magnitude is zero. Component bias is $\mathbf d_i-\hat{\mathbf d}_i$. Retention is retained candidates divided by all candidates on the sampled grid (or separately inside the interior ROI). The **median EPE on retained points can decrease merely because difficult vectors were rejected**. Always report EPE together with retention, and ideally evaluate unfiltered output on the common complete grid or compare methods at matched retention.

The Rankine script's `estimate_center_from_vectors` uses $\mathbf d\cdot(\mathbf x-\mathbf c)=0$, which is exact for an infinitesimal tangent vector but **not for a finite rotation chord**. A finite rotation instead yields $\mathbf d\cdot(\mathbf x-\mathbf c)=-\|\mathbf d\|^2/2$ for perfect circular orbits. Therefore its center-error metric may contain a systematic bias; correct and validate this before citing it scientifically. Similarly, its `recovered_core_radius_px` is a coarse bin-center proxy, not a fitted core radius.

## Publication-grade experimental matrix

At minimum use identical input frames and seeds across methods, then vary one factor at a time: explicit solver with information-aware ridge off/on; spatial prior off/on; image warp vs coefficient warp; consistency mask off/on; confidence smoothing off/on; SVD off/on. Fix all other parameters. Vary translation magnitude and direction (including zero), vortex core radius/angular displacement, seed, noise and particle density. Record median/mean/p95 EPE, component bias, angular error, retained fraction, spatial maps, timing, input hashes, code commit, and environment. Don't compare unrelated code revisions/settings as though they were a single-factor ablation.
