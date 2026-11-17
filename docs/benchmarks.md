# Benchmark generation and validation

## Ground-truth conventions

Optical flow is a finite **inter-frame displacement**, not necessarily the instantaneous Eulerian velocity at the first frame. In a synthetic test, ground truth should use the same temporal correspondence and sampling locations as the estimator. With image coordinates $x$ right and $y$ down, the measured displacement is $\mathbf d=(u,v)$ in `px/frame pair` (typically written `px/frame` for consecutive frames). To convert to `mm/s`, multiply by `pixel_size_mm / frame_interval_s`.

## Submitted uniform generator

The original `benchmarks/generators/generate_uniform_flow_test.py` creates 512×512 images, 100 frames, ~4000 visible tracer particles, Gaussian particle width 1.1 px, approximate peak 220 and random seed 42 **in its submitted default configuration**. The file currently sets `(TRUE_U, TRUE_V)=(1,0)` px/frame; do **not** mistake this for the separately submitted historical `(8,-20)` run. Every particle center evolves as

$$x_i(t)=x_i(0)+t u_0,\qquad y_i(t)=y_i(0)+t v_0.$$

Subpixel positions are bilinearly splatted onto four pixels, Gaussian blurred, fixed-intensity scaled, clipped to 8-bit, and saved as TIFF. Particles are initially seeded beyond the image extent to support entries/exits. The generator writes `true_vector_values.npz` (arrays `u_gt`, `v_gt`, and scalars `true_u`, `true_v`, `true_speed`) plus `parameters.json`. Its final console message currently **incorrectly says `ground_truth.npz`**—the actual saved file name is correct. The original generator also uses a hard-coded local Windows output directory; edit it before running.

The new `generate_mini_uniform.py` is only a **small, portable demonstrator** that uses the same particle-rendering concepts and writes compatible ground truth. It does not recreate the historical archived uniform data.

## Rankine vortex benchmark

A Rankine vortex has piecewise tangential speed

$$
v_{\theta}(r)=\begin{cases}\Omega r,& r\le R,\\ \Omega R^2/r,&r>R,\end{cases}
\qquad \Omega=V_R/R,
$$

with $r=\sqrt{(x-c_x)^2+(y-c_y)^2}$. A tracer orbit has angular increment $\Delta\theta=\Omega\Delta t$ inside the core and $\Delta\theta=\Omega R^2\Delta t/r^2$ outside. For an exact finite-displacement benchmark, rotate the initial point by $\Delta\theta$, then define **ground truth as the final minus initial Cartesian position** (the chord), not the local tangent multiplied by $\Delta t$. This is particularly important at large angular displacements. These equations describe a suitable Rankine benchmark; **the actual Rankine generator was not provided**, so its implementation-specific details are not yet verified.

The supplied Rankine validator expects `true_vector_values.npz` keys `u_gt`, `v_gt`, `speed_gt`, `radius_gt`, `radial_gt`, `tangential_gt`, `continuous_tangential_speed_gt`, `continuous_vorticity_gt`, `continuous_divergence_gt`, `center_x`, `center_y`, `core_radius_px`, `boundary_tangential_speed_px`, `omega_core_rad_per_frame`. The historical run's name indicates `R80`, `vb3`, and `(cx,cy)=(255.5,255.5)`; **filenames alone are not adequate proof of the generating parameter values**.

## Validation tools (as supplied)

The `benchmarks/evaluators/` scripts read `validation_*.npz`, access ground truth at sampled `x`,`y`, and report per-pair / summary metrics. Both scripts default to a 24-pixel interior exclusion. The Rankine script additionally uses a minimum true speed of 0.25 px/frame for angular/relative error, 14 radial bins, radial/tangential bias, center error and a binned core-radius proxy. **Update `SYNTHETIC_TEST_DIR` and `PIPELINE_OUTPUT_DIR`** in each original evaluator before use; they are hard-coded to local Windows directories.

### Metrics

Endpoint error (EPE) compares estimated and independently known vectors at the same sample point:

$$\mathrm{EPE}_i=\sqrt{(u_i-\hat u_i)^2+(v_i-\hat v_i)^2}.$$

The two vectors' angular error is $\theta_i=\cos^{-1}[(\mathbf d_i\cdot\hat{\mathbf d}_i)/(\|\mathbf d_i\|\|\hat{\mathbf d}_i\|)]$; it is undefined if either magnitude is zero. Component bias is $\mathbf d_i-\hat{\mathbf d}_i$. Retention is retained candidates divided by all candidates on the sampled grid (or separately inside the interior ROI). The **median EPE on retained points can decrease merely because difficult vectors were rejected**. Always report EPE together with retention, and ideally evaluate unfiltered output on the common complete grid or compare methods at matched retention.

The Rankine script's `estimate_center_from_vectors` uses $\mathbf d\cdot(\mathbf x-\mathbf c)=0$, which is exact for an infinitesimal tangent vector but **not for a finite rotation chord**. A finite rotation instead yields $\mathbf d\cdot(\mathbf x-\mathbf c)=-\|\mathbf d\|^2/2$ for perfect circular orbits. Therefore its center-error metric may contain a systematic bias; correct and validate this before citing it scientifically. Similarly, its `recovered_core_radius_px` is a coarse bin-center proxy, not a fitted core radius.

## Publication-grade experimental matrix

At minimum use identical input frames and seeds across methods, then vary one factor at a time: explicit solver with information-aware ridge off/on; spatial prior off/on; image warp vs coefficient warp; consistency mask off/on; confidence smoothing off/on; SVD off/on. Fix all other parameters. Vary translation magnitude and direction (including zero), vortex core radius/angular displacement, seed, noise and particle density. Record median/mean/p95 EPE, component bias, angular error, retained fraction, spatial maps, timing, input hashes, code commit, and environment. Don't compare unrelated code revisions/settings as though they were a single-factor ablation.
