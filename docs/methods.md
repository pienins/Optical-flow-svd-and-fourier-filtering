# Methods — explicit Farnebäck optical flow with two-way consistency (2WCF-SVD)

**Scope.** This document describes the numerical method implemented in `two_wcf/` on the `modular-2wcf` branch. Equations describe what the code computes. They do not by themselves establish physical accuracy, accuracy against ground truth, or novelty; validation status is recorded in `benchmarks.md` and `status_and_limitations.md`. Software execution (parallelism, streaming, file layout) is described in `code_map.md`, and output arrays and sign conventions in `output_formats.md`.

Each section separates the **standard method** (Farnebäck polynomial-expansion flow, robust statistics, forward–backward checks) from **project-specific modifications**. Section 8 lists the design choices that still need experimental justification.

## 0. Conventions and notation

- Frames $I_1 = I_t$ and $I_2 = I_{t+1}$ are consecutive images of one sequence. Pixel coordinates $\mathbf x=(x,y)^\top$ have $x$ increasing to the right and $y$ increasing **downward**.
- The forward displacement $\mathbf d(\mathbf x)=(u,v)^\top$ is defined by $I_1(\mathbf x)\approx I_2(\mathbf x+\mathbf d(\mathbf x))$. Positive $u$ is image-right, positive $v$ is image-down.
- Units are **pixels per frame pair** (px/frame for consecutive frames). The flow is a finite displacement between two rendered frames, not an instantaneous velocity.
- The backward field $\mathbf d_{21}$ is computed independently with the frames swapped.

| Symbol | Meaning | JSON key |
|---|---|---|
| $\mathcal K$ | removed temporal SVD modes (zero-based) | `KillIDs`, enabled by `USE_SVD` |
| $n_p$, $\sigma_p$ | polynomial-fit support width and Gaussian applicability | `PolyFitKernelSize`, `GaussKernelStdev` |
| $n_w$ | motion-integration window | `WindowSize`, type from `FilterFlags` |
| $s$, $L_{\max}$, $n_{\min}$ | pyramid scale, maximum level count, minimum level side | `PyramidDownsampleFactor`, `PyramidLevels`, `FarnebackPyramidMinSize` |
| $N$ | iterations per pyramid level | `IterationCount` |
| $\delta_{\max}$ | per-iteration increment cap | `ExplicitMaxResidualIncrement` |
| $r$ | baseline diagonal regularization | `ExplicitFarnebackRegularization` |
| $f$, $k_p$ | information ridge fraction and prior window | `InformationRidgeFraction`, `InformationPriorWindow` |
| $k$, $\alpha$, $\tau_0$ | MAD multiplier, relative consistency weight, fallback threshold | `HardThresholdK`, `RelativeConsistencyAlpha`, `ConsistencyErrorThreshold` |
| $k_c$, $c_{\min}$ | confidence MAD multiplier and minimum confidence | `ConfidenceSigma`, `MinConfidence` |

## 1. Preprocessing: temporal SVD and intensity normalization

### 1.1 Optional temporal SVD mode removal

All $T$ frames, each $H\times W$, are stacked as rows of $X\in\mathbb R^{T\times HW}$ (float32). The thin SVD $X=U\Sigma V^\top$ is computed **without subtracting the temporal mean**, so mode 0 mainly represents the static (time-averaged) image. The selected modes are subtracted:

$$
X_{\text{filt}} = X-\sum_{k\in\mathcal K}\sigma_k\,\mathbf u_k\mathbf v_k^\top .
$$

`KillIDs` are zero-based. Indices outside $[0,\operatorname{rank}-1]$ are skipped with a warning. Repeated indices are rejected when the configuration is loaded, because a repeated mode would be subtracted more than once.

**Mode selection is not part of the pipeline.** `KillIDs` are set in the configuration. Frequency analysis of temporal modes is done separately in `Artifact_mode_detection/`. **No Fourier-domain filtering is applied to the images**, despite the repository's historical name.

### 1.2 Sequence-global normalization and quantization

After optional SVD, one minimum and maximum are taken over **all frames** of the sequence. Each frame is mapped to 8 bits by truncation:

$$
I^{(8)}_t=\left\lfloor 255\,\operatorname{clip}\!\left(\frac{I_t-I_{\min}}{I_{\max}-I_{\min}},\,0,\,1\right)\right\rfloor .
$$

If $I_{\max}=I_{\min}$ the division is skipped. The solver converts the 8-bit images to float and divides by 255, so polynomial fitting operates on $[0,1]$ intensities with 256 discrete levels. A global (not per-frame) range preserves relative brightness between frames, but one extreme pixel anywhere in the sequence compresses the contrast of every frame.

**Implementation:** `image_io.svd_filter_images`, `image_io.load_images_without_svd`, `image_io.normalize_frame`, `pipeline.run`.

## 2. Local polynomial expansion (standard Farnebäck)

Around each pixel, intensity is approximated by a quadratic in local coordinates $\boldsymbol\xi=(\xi_x,\xi_y)^\top$:

$$
I(\mathbf x+\boldsymbol\xi)\simeq c+p_x\xi_x+p_y\xi_y+q_{xx}\xi_x^2+q_{xy}\xi_x\xi_y+q_{yy}\xi_y^2
=\boldsymbol\xi^\top A\boldsymbol\xi+\mathbf b^\top\boldsymbol\xi+c,
$$

$$
A=\begin{bmatrix}q_{xx}&q_{xy}/2\\ q_{xy}/2&q_{yy}\end{bmatrix},\qquad \mathbf b=\begin{bmatrix}p_x\\ p_y\end{bmatrix}.
$$

The six coefficients are the weighted least-squares fit over an $n_p\times n_p$ grid ($\xi_x,\xi_y\in\{-\lfloor n_p/2\rfloor,\dots,\lfloor n_p/2\rfloor\}$) with applicability $w(\boldsymbol\xi)=\exp[-\|\boldsymbol\xi\|^2/(2\sigma_p^2)]$. With design matrix $B$ (columns $1,\xi_x,\xi_y,\xi_x^2,\xi_x\xi_y,\xi_y^2$) and $W=\operatorname{diag}(w)$, the coefficient operator $(B^\top WB)^{+}B^\top W$ is precomputed once. Its six rows become six correlation kernels applied to the whole image (reflect-101 borders). Certainty is uniform, so this is ordinary weighted least squares, not normalized convolution with a confidence map.

$n_p$ is the **full width** of the fit support and must be odd and $\ge 3$. The polynomial support is unrelated to the motion-integration window $n_w$ in Section 3.

**Implementation:** `solver.make_quadratic_fit_kernels`, `solver.quadratic_polynomial_expansion`.

## 3. Displacement estimation (standard Farnebäck core)

### 3.1 Local motion constraint

If $I_1(\mathbf x)=I_2(\mathbf x+\mathbf d)$ exactly, substituting the quadratic model of $I_1$ gives $A_2=A_1$ and $\mathbf b_2=\mathbf b_1-2A_1\mathbf d$. With real data the code uses the averaged quantities

$$
A=\tfrac12(A_1+A_2),\qquad \Delta\mathbf b=\tfrac12(\mathbf b_1-\mathbf b_2),
\qquad A\,\mathbf d\approx\Delta\mathbf b .
$$

### 3.2 Neighborhood least squares

One pixel's $2\times2$ constraint is generally ill-posed, so $\mathbf d$ is assumed constant over a window $\mathcal N(\mathbf x)$ and the weighted error is minimized:

$$
\varepsilon(\mathbf d)=\sum_{\mathbf y\in\mathcal N(\mathbf x)}\omega(\mathbf y-\mathbf x)\,\big\|A(\mathbf y)\mathbf d-\Delta\mathbf b(\mathbf y)\big\|^2
\;\Longrightarrow\;
G\,\mathbf d=\mathbf h,
$$

$$
G=\sum\omega\,A^\top A,\qquad \mathbf h=\sum\omega\,A^\top\Delta\mathbf b .
$$

The window $\omega$ is an **unnormalized box** of side $n_w$ (`FilterFlags = 0`) or a Gaussian (`FilterFlags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN`). The Gaussian uses side $n_w$ (increased by 1 if even) and OpenCV's default $\sigma=0.3\,((n_w-1)/2-1)+0.8$. Window sums use reflect-101 borders. Other flag bits are rejected. The $2\times2$ system is solved in closed form. Pixels with $|\det|\le10^{-12}$ or non-finite terms receive no update.

$G$ is symmetric positive semidefinite. Its eigenvalues $\lambda_{\min}\le\lambda_{\max}$ measure how strongly the windowed image structure constrains each displacement direction (the aperture problem corresponds to $\lambda_{\min}\approx0$). Section 4 uses this.

### 3.3 Iterative refinement: two warp modes

Given a current estimate $\mathbf d_0$, each iteration solves for a correction. `FarnebackWarpMode` selects how frame 2 is brought to $\mathbf x+\mathbf d_0$:

- **`coefficients` (OpenCV-style; current default).** Frame-2 coefficients are fitted **once per level** and bilinearly sampled at $\mathbf x+\mathbf d_0$. Because they are not refitted on a warped image, the constraint for the **absolute** displacement is

  $$A\,\mathbf d=\Delta\mathbf b+A\,\mathbf d_0,$$

  and the solver returns $\mathbf d-\mathbf d_0$ as the increment. Singular pixels keep $\mathbf d_0$. With `CoefficientWarpOpenCVBorders = true` the OpenCV boundary rules are reproduced:
  - Samples are valid only if $0\le x+u<W-1$ and $0\le y+v<H-1$.
  - At invalid samples, frame-1 curvature is used ($A=A_1$) and frame-2 linear terms are zeroed ($\Delta\mathbf b=\tfrac12\mathbf b_1$).
  - $A$ and $\Delta\mathbf b$ are multiplied by OpenCV's 5-pixel edge taper (weights 0.14, 0.14, 0.4472, 0.4472, 0.4472 from the edge inward, separable in $x$ and $y$) **before** $G$ and $\mathbf h$ are formed.

  With `false`, coefficients are sampled with reflect-101 everywhere, with no validity mask or taper. This is an ablation setting.
- **`image` (legacy).** Frame 2 is bilinearly warped by $\mathbf d_0$ (reflect-101), its coefficients are refitted every iteration, and a **residual** correction $\boldsymbol\delta$ is solved from $A\boldsymbol\delta=\Delta\mathbf b$.

In both modes $\mathbf d\leftarrow\mathbf d+\boldsymbol\delta$ after each of the $N$ iterations. Frame-1 coefficients are fitted once per level.

### 3.4 Coarse-to-fine pyramid

Level $\ell$ has scale $s^\ell$ and size $\operatorname{round}(Ws^\ell)\times\operatorname{round}(Hs^\ell)$. Each level is built **directly from the full-resolution 8-bit frame**: Gaussian pre-smoothing with $\sigma_\ell=\tfrac12(s^{-\ell}-1)$ and kernel width $\operatorname{round}(5\sigma_\ell)$ forced odd, followed by bilinear resizing. Levels are added while the smaller side stays $\ge n_{\min}$, up to $L_{\max}$ levels **including** full resolution. `PyramidLevels` is therefore a maximum, not a guarantee. For a 512×512 image with $s=0.5$ and $n_{\min}=32$ the actual widths are 512, 256, 128, 64 and 32 (five levels).

Processing starts from zero flow at the coarsest level. Between levels the flow is resized bilinearly and each component is multiplied by the ratio of the new to the old grid width (for $u$) or height (for $v$). This compensates exactly for rounded level sizes.

**Implementation:** `solver.build_farneback_visualization_pyramid` (despite its name, this is the solver's own pyramid), `solver.sample_farneback_coefficients`, `solver.warp_second_frame`, `solver.farneback_residual_increment`, `solver.resize_flow_field`, `solver.calc_optical_flow_farneback_explicit`.

## 4. Project-specific modifications to the solver

These steps are **not** part of standard Farnebäck. All are optional or tunable, and none has yet been justified by an ablation against independent ground truth (Section 8).

### 4.1 Per-iteration increment cap

If $\delta_{\max}$ is set, every increment is rescaled so that $\|\boldsymbol\delta\|\le\delta_{\max}$, measured in pixels of the current level. In `coefficients` mode the cap applies to the change from $\mathbf d_0$, after any regularization.

**Consequence: a hard bound on recoverable displacement.** Starting from zero flow, the full-resolution displacement cannot exceed approximately

$$
\|\mathbf d\|_{\max}\approx N\,\delta_{\max}\sum_{\ell=0}^{L-1}\frac{W}{W_\ell}\;\approx\;N\,\delta_{\max}\sum_{\ell=0}^{L-1}s^{-\ell},
$$

where $L$ is the actual number of levels. For the 512×512, $s=0.5$ example with $N=2$ and $\delta_{\max}=0.5$ px, the bound is about 31 px. With $s=0.45$ (levels 512, 230, 104, 47) it falls to about 19 px. Any synthetic benchmark whose true displacement approaches this bound is testing the cap, not the estimator.

### 4.2 Baseline diagonal regularization

When the information-aware solver is disabled, the system becomes

$$
\big(G+\lambda_b I\big)\,\mathbf d=\mathbf h+\lambda_b\,\mathbf d_0,
\qquad \lambda_b(\mathbf x)=r\,\big(\operatorname{tr}G(\mathbf x)+1\big)
$$

in `coefficients` mode. In `image` mode the right-hand side is unchanged, which is the same penalty written for the residual. Both forms minimize $\varepsilon(\mathbf d)+\lambda_b\|\mathbf d-\mathbf d_0\|^2$, i.e. they damp the update. This is **not** OpenCV's regularization (Section 7). Because of the "+1", its relative strength depends on intensity scaling and on the unnormalized window sum.

### 4.3 Information-aware ridge

With `UseInformationAwareSolver = true`, the raw (unregularized) eigenvalues of $G$ are computed at every pixel. One **scalar** ridge per solve is derived from them:

$$
\lambda=\max\!\big(10^{-12},\;f\,P_{75}\{\lambda_{\max}(G)\}\big),
$$

where the 75th percentile is taken over all finite pixels of the current level. It is recomputed for every iteration, level, frame pair and flow direction. Referencing a percentile makes $\lambda$ independent of intensity scale and window normalization. The solved system is

$$
(G+\lambda I)\,\mathbf d=\mathbf h+\lambda\,\mathbf p,
\qquad\text{equivalently}\qquad
\mathbf d=\arg\min_{\mathbf d}\;\varepsilon(\mathbf d)+\lambda\|\mathbf d-\mathbf p\|^2,
$$

with prior $\mathbf p=\mathbf d_0$ unless the spatial prior (4.4) is enabled. In `image` mode the same objective is written for the residual, with right-hand side $\mathbf h+\lambda(\mathbf p-\mathbf d_0)$.

**Interpretation.** In the eigenbasis of $G$, along a direction with eigenvalue $\mu$, the solution moves the fraction $\mu/(\mu+\lambda)$ of the way from $\mathbf p$ toward the data-only solution $G^{-1}\mathbf h$. Weakly constrained directions ($\mu\ll\lambda$) stay near the prior, as intended. Because $\lambda$ is referenced to the 75th percentile of $\lambda_{\max}$, choosing $f\ge1$ makes $\lambda$ exceed $\lambda_{\max}$ at about 75 % or more of pixels. There, even the best-constrained direction recovers less than half of the data correction per iteration. Large $f$ combined with few iterations per level therefore damps convergence as well as stabilizing weak directions. A uniform-translation benchmark confirms this for the shipped $f=2$: displacement was underestimated by about 20 % (`status_and_limitations.md` §1).

### 4.4 Information-weighted spatial prior

With `UseInformationSpatialPrior = true` and a current estimate available, the prior becomes a trust-weighted local mean of the current flow:

$$
w(\mathbf x)=\frac{\lambda_{\min}(\mathbf x)}{\lambda_{\min}(\mathbf x)+\lambda},\qquad
\mathbf p(\mathbf x)=\frac{\sum_{\mathbf y\in\mathcal B_{k}(\mathbf x)}w(\mathbf y)\,\mathbf d_0(\mathbf y)}{\sum_{\mathbf y\in\mathcal B_k(\mathbf x)}w(\mathbf y)} .
$$

$\mathcal B_k$ is a box of side $k=\max(3,k_p)$ made odd, which includes the centre pixel. Non-finite weights are set to 0. If $\sum w\le10^{-8}$ the prior falls back to $\mathbf d_0(\mathbf x)$. Weakly constrained pixels are thereby pulled toward the flow of better-constrained neighbours rather than toward their own previous value.

**Caution.** Better conditioning, smaller forward–backward error or a smoother field do **not** establish lower endpoint error against ground truth. A ridge or prior can make forward and backward estimates agree with each other while both are biased toward the prior. These modifications must be evaluated by ablation on benchmarks with known displacement, reporting accuracy together with retention.

**Implementation:** `solver.farneback_residual_increment` (both regularization branches and the information diagnostics).

## 5. Two-way consistency and hard selection

### 5.1 Forward–backward error (standard check)

The forward field $\mathbf d_{12}$ and backward field $\mathbf d_{21}$ are computed by two independent solver calls. The backward field is sampled at the forward endpoint by bilinear interpolation (reflect border):

$$
\mathbf r(\mathbf x)=\mathbf d_{12}(\mathbf x)+\mathbf d_{21}\big(\mathbf x+\mathbf d_{12}(\mathbf x)\big),\qquad e(\mathbf x)=\|\mathbf r(\mathbf x)\|_2 .
$$

For a perfectly consistent pair $e=0$. Endpoints that leave the image are sampled from the reflected field, so $e$ is not meaningful for inflow/outflow regions or near borders.

### 5.2 Adaptive threshold (robust statistics)

Over all finite pixels of the frame pair the code computes the median $m$ and scaled MAD $D=1.4826\,\operatorname{median}(|e-m|)$. The absolute tolerance is

$$
\tau=m+k\,D .
$$

If $m$ or $D$ is not finite, $\tau=\tau_0$ (`ConsistencyErrorThreshold`). $\tau_0$ is therefore a fallback, **not** an independent cutoff. In practice it has no effect on results: the fallback is reached only when no pixel has a finite error, and then every vector is rejected anyway. (Non-finite flow would also stall `cv2.remap` before this point.) A median-of-means of $e$ (groups of `MedianOfMeansGroupSize` in raster order; remainder discarded) is recorded for quality control only and does not enter any decision.

### 5.3 Hybrid relative/absolute mask

A pixel is consistent when

$$
e(\mathbf x)^2\le\alpha\Big(\|\mathbf d_{12}(\mathbf x)\|^2+\big\|\mathbf d_{21}(\mathbf x+\mathbf d_{12})\big\|^2\Big)+\tau^2 .
$$

This has the same form as the widely used forward–backward occlusion test (e.g. Sundaram, Brox & Keutzer, 2010, with fixed constants 0.01 and 0.5 px²). The project-specific parts are the **frame-adaptive** $\tau$ and the configurable $\alpha$. The relative term tolerates larger absolute inconsistency for larger displacements. A large $\alpha$ therefore weakens the check most where motion is fastest.

### 5.4 Sampling and output in hard mode

The error and mask are computed on the dense grid, then sampled every `ArrowDensityStep` pixels starting at offset `ArrowDensityStep // 2` in both directions. In hard mode (`UseConfidenceSmoothing = false`) the exported vectors are the **unmodified forward flow** at sampled pixels that pass the mask. Rejected positions are recorded separately. Confidence (Section 6) is computed only as a diagnostic.

**Implementation:** `consistency.compute_flow_pair`, `consistency.median_of_means`.

## 6. Soft confidence mode (optional)

With `UseConfidenceSmoothing = true`, a confidence is assigned to every pixel:

$$
c(\mathbf x)=\exp\!\big[-\big(e(\mathbf x)/\sigma(\mathbf x)\big)^2\big]\cdot\mathbb 1_{\text{mask}}(\mathbf x).
$$

- **Global scale (default):** $\sigma=m+k_c D$ for the frame pair. If the statistics are not finite, $\sigma$ falls back to `MinimumConfidenceSigma` (the same degenerate case as in 5.2).
- **Local scale** (`UseLocalConfidenceSigma = true`): $\sigma(\mathbf x)=\tilde m(\mathbf x)+k_c\,\tilde D(\mathbf x)$. Here $\tilde m$ is a median filter of $e$ over `LocalSigmaKernelSize` (odd, $\ge3$, nearest-edge mode), and
  $$\tilde D(\mathbf x)=1.4826\;\operatorname*{median}_{\mathbf y\in\mathcal N(\mathbf x)}\big|e(\mathbf y)-\tilde m(\mathbf y)\big|.$$
  Each deviation is taken from **its own** neighbourhood median $\tilde m(\mathbf y)$, not from $\tilde m(\mathbf x)$. This approximates the true local MAD.
- In both cases $\sigma$ is bounded below by `MinimumConfidenceSigma`.

The confidence is multiplied by the hard mask **before** smoothing, so inconsistent vectors never contribute. The flow is then replaced by a confidence-weighted box average over `ConfidenceKernelSize` (odd):

$$
\tilde u(\mathbf x)=\frac{\sum_{\mathcal N(\mathbf x)}c\,u}{\sum_{\mathcal N(\mathbf x)}c+10^{-6}},\qquad
\tilde v(\mathbf x)=\frac{\sum_{\mathcal N(\mathbf x)}c\,v}{\sum_{\mathcal N(\mathbf x)}c+10^{-6}} .
$$

A sampled vector is retained only if it passes the hybrid mask **and** its own pointwise confidence satisfies $c\ge c_{\min}$. Soft mode is therefore a stricter selection than hard mode followed by spatial smoothing. It does not replace the hard mask. Smoothing reduces noise but also spatial resolution, and it biases sharp gradients such as vortex cores and shear layers. Its effect must be judged against ground truth, not consistency.

**Implementation:** `consistency.calculate_local_sigma`, `consistency.confidence_weighted_smoothing`, `consistency.compute_flow_pair`.

## 7. Relation to OpenCV `calcOpticalFlowFarneback`

The solver does not call OpenCV's Farnebäck routine. It follows OpenCV's algorithm closely in `coefficients` mode with OpenCV borders, but it is **not numerically identical**, and parameter values are **not interchangeable**:

| Aspect | OpenCV (`optflowgf.cpp`, 4.x) | This implementation |
|---|---|---|
| Polynomial support | `poly_n` is a half-width: support $(2\,\texttt{poly\_n}+1)^2$; only 5 or 7 accepted (11×11 or 15×15) | `PolyFitKernelSize` is the full width ($n_p\times n_p$, any odd $\ge3$) |
| Polynomial fit | separable closed form, Gaussian applicability | full $6\times6$ weighted least squares via pseudoinverse (same model) |
| Pyramid level count | `levels` = number of reductions *beyond* full resolution; minimum side fixed at 32 | `PyramidLevels` = total levels *including* full resolution; minimum side configurable |
| Full-resolution level | also blurred (3×3 kernel, $\sigma=0.8$ from OpenCV's default) | not smoothed |
| Pre-smoothing kernel | at least 3×3 | may be 1×1 (no smoothing) at very mild scales |
| Flow upsampling | multiply by $1/s$ | multiply by the exact width/height ratio |
| Gaussian window | $\sigma=0.3\lfloor n_w/2\rfloor$ | $\sigma=0.3((n_w-1)/2-1)+0.8$ |
| Box window | normalized sum | unnormalized sum (ratio unaffected unless regularized) |
| Regularization | $10^{-3}$ added to $\det G$ (0–255 intensities) | $r(\operatorname{tr}G+1)$ or information ridge on the diagonal, $[0,1]$ intensities |
| Increment cap | none | optional $\delta_{\max}$ (Section 4.1) |
| Coefficient sampling | bilinear with exact float weights | `cv2.remap` bilinear, which quantizes interpolation weights |

Results from OpenCV runs and from this solver should be compared only after mapping parameters explicitly (for example, OpenCV `levels = PyramidLevels − 1`).

## 8. Design choices requiring experimental justification

The table separates what the code does from what has been shown to help. "Ablation needed" means no controlled comparison against ground truth is recorded in this repository.

| Choice | Alternative to compare | Status |
|---|---|---|
| `coefficients` vs `image` warp mode | each against the other | ablation needed |
| OpenCV border fallback and taper | `CoefficientWarpOpenCVBorders = false` | ablation needed |
| Increment cap $\delta_{\max}$ | no cap; cap relative to displacement bound (4.1) | ablation needed; check bound vs benchmark speeds |
| Information ridge, fraction $f$, 75th percentile reference | baseline regularization; $f\in\{0.01,0.1,1,2\}$ | ablation needed |
| Information spatial prior and window | prior off | ablation needed |
| Hybrid mask constants $k$, $\alpha$ | pure absolute ($\alpha=0$); pure relative | ablation needed, with retention reported |
| Soft mode and confidence parameters | hard mode | ablation needed; resolution loss at gradients |
| Global vs local confidence scale | each against the other | ablation needed |
| Temporal SVD mode removal | `USE_SVD = false`; different $\mathcal K$ | experimental data only; effect on accuracy unknown |

Every ablation must report endpoint error **and** retention on the same vectors. A setting that lowers retained-vector endpoint error by rejecting difficult regions has not improved the estimator.
