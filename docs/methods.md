# Methods — current modular explicit 2WCF implementation

**Scope.** This documents the code under `two_wcf/` in the uploaded modular ZIP, not every historical solver variant. Symbols are chosen to match the numerical calculations. Equations describe the implemented operations; they do not themselves establish physical accuracy or algorithmic novelty.

## 1. Image input, intensity scaling and optional temporal SVD

The pipeline reads a grayscale TIFF sequence $I_t(x,y)$, optionally applies temporal SVD, and then computes one **sequence-global** min/max. Each frame is normalized to $[0,255]$ and converted to `uint8` for optical-flow input:

$$
I_t^{(8)}=\operatorname{uint8}\!\left(255\operatorname{clip}\left(\frac{I_t-I_{\min}}{I_{\max}-I_{\min}},0,1\right)\right).
$$

For SVD, flatten each frame into a row of a matrix $X\in\mathbb{R}^{T\times HW}$. Compute $X=U\Sigma V^\top$ and subtract selected mode contributions $\sum_{k\in\mathcal K}\sigma_k\mathbf u_k\mathbf v_k^\top$. A `KillIDs` entry is **zero-based**. Out-of-range indices are skipped with a warning. The code plots FFT spectra of temporal modes only when `do_plots=True`; it **does not implement a separate frequency-domain filtering operation**.

**Functions:** `image_io.svd_filter_images`, `image_io.load_images_without_svd`, `image_io.normalize_frame`.

## 2. Local polynomial representation of intensity

At each pixel, approximate intensity over a neighborhood by a second-degree polynomial in local coordinates $\boldsymbol\xi=(\xi_x,\xi_y)^\top$:

$$
I(\mathbf{x}+\boldsymbol\xi)\simeq c+p_x\xi_x+p_y\xi_y+q_{xx}\xi_x^2+q_{xy}\xi_x\xi_y+q_{yy}\xi_y^2
=\boldsymbol\xi^\top A\boldsymbol\xi+\mathbf b^\top\boldsymbol\xi+c,
$$

$$
A=\begin{bmatrix}q_{xx}&q_{xy}/2\\q_{xy}/2&q_{yy}\end{bmatrix},\qquad \mathbf b=(p_x,p_y)^\top.
$$

The six coefficients are obtained with Gaussian-weighted least squares on a `PolyFitKernelSize` × `PolyFitKernelSize` grid with applicability $w(\boldsymbol\xi)=\exp[-\|\boldsymbol\xi\|^2/(2\sigma_p^2)]$. In code, the precomputed pseudoinverse of the design system becomes six correlation kernels applied to every image pixel. `GaussKernelStdev` controls $\sigma_p$. These **polynomial fit neighborhoods are different from** the subsequent motion-integration window.

**Functions:** `solver.make_quadratic_fit_kernels`, `solver.quadratic_polynomial_expansion`.

## 3. Motion equations and iterative coarse-to-fine solution

The forward displacement field $\mathbf d_{12}(\mathbf x)=(u,v)^\top$ follows $I_1(\mathbf x)\approx I_2(\mathbf x+\mathbf d_{12}(\mathbf x))$. Reverse flow is computed independently with the frames swapped. Flow is **displacement per frame pair**, represented in image coordinates where positive $x$ is right and positive $y$ is downward.

From the local quadratic models, the code constructs

$$
\bar A=\tfrac12(A_1+A_2),\qquad \Delta\mathbf b=\tfrac12(\mathbf b_1-\mathbf b_2),
$$

and combines neighboring estimates over `WindowSize`:

$$
G(\mathbf x)=\sum_{\mathbf y\in\mathcal N(\mathbf x)}\bar A(\mathbf y)^\top\bar A(\mathbf y),\quad
\mathbf h(\mathbf x)=\sum_{\mathbf y\in\mathcal N(\mathbf x)}\bar A(\mathbf y)^\top\Delta\mathbf b(\mathbf y).
$$

Sums here represent the box or Gaussian integration prescribed by `FilterFlags`. The 2×2 system $G\boldsymbol\delta=\mathbf h$ is solved analytically with determinant checks. The exact right-hand side depends on the warp mode:

- **`image`:** warp frame 2 by the current forward flow $\mathbf d_0$, refit its quadratic coefficients each iteration, and solve a residual correction $\boldsymbol\delta$.
- **`coefficients`:** fit frame-2 coefficients once per level, sample those maps at $\mathbf x+\mathbf d_0$, add $\bar A\mathbf d_0$ to $\Delta\mathbf b$, then solve for an **absolute displacement**, returning its difference from $\mathbf d_0$ as the iterative increment. Optional boundary validity fallback and five-pixel taper are controlled by `CoefficientWarpOpenCVBorders`. This is **OpenCV-style**, not numerically identical to OpenCV's C++ solver.

Each increment is norm-clipped to `ExplicitMaxResidualIncrement` if the cap is enabled. The solver uses an image pyramid built directly from the original image at scale $s^\ell$, with level-dependent Gaussian smoothing before resizing. Processing proceeds from the coarsest valid level to full resolution. On upsampling, each flow component is multiplied by the corresponding width/height ratio.

`PyramidLevels` is a **maximum**, not a guarantee. For a 512×512 image, scale 0.5, and minimum side 32, the actual levels have widths 512, 256, 128, 64, and 32 (five levels). `IterationCount` iterations run at each actual level.

**Functions:** `solver.build_farneback_visualization_pyramid`, `solver.sample_farneback_coefficients`, `solver.farneback_residual_increment`, `solver.resize_flow_field`, `solver.calc_optical_flow_farneback_explicit`.

## 4. Information-aware local motion regularization

The raw normal matrix $G$ is symmetric positive semidefinite. Its eigenvalues $\lambda_{\min},\lambda_{\max}$ indicate how strongly the local image supports independent displacement components. When `UseInformationAwareSolver=true`, the implementation sets

$$
\lambda_{\mathrm{ridge}}=\max\!\left(10^{-12},f\,\mathrm{P}_{75}(\lambda_{\max}(G))\right),
$$

where $f=\texttt{InformationRidgeFraction}$ and the percentile is computed on the current pyramid level. The system matrix becomes $G+\lambda_{\mathrm{ridge}}I$. If the spatial prior is disabled, the right-hand side is centered on the current flow for an absolute solve; in residual mode it remains centered on zero increment.

When `UseInformationSpatialPrior=true` and a current estimate exists, a trust weight

$$
w(\mathbf x)=\frac{\lambda_{\min}(\mathbf x)}{\lambda_{\min}(\mathbf x)+\lambda_{\mathrm{ridge}}}
$$

is used for a box-supported local weighted mean of the **current displacement** over `InformationPriorWindow`. The ridge right-hand side favors that local prior (or the corresponding residual to current flow). If no finite weight is available, the prior defaults to the current pixel's displacement. The code's `UseInformationAwareSolver=false` branch instead uses its small `ExplicitFarnebackRegularization` multiplied by `(trace(G)+1)`; this is not a completely unregularized inversion.

**Caution:** Better conditioning, smaller reported forward/backward error, or a smoother field do **not** establish lower ground-truth endpoint error. The ridge and prior must be tested in ablations.

## 5. Two-way consistency and hard selection

Estimate both forward $\mathbf f_{12}(\mathbf x)$ and backward $\mathbf f_{21}(\mathbf y)$ displacement. Sample backward flow at the forward endpoint $\mathbf y=\mathbf x+\mathbf f_{12}(\mathbf x)$:

$$
\mathbf r(\mathbf x)=\mathbf f_{12}(\mathbf x)+\mathbf f_{21}(\mathbf x+\mathbf f_{12}(\mathbf x)),\quad e(\mathbf x)=\|\mathbf r(\mathbf x)\|_2.
$$

The backward field is bilinearly remapped with a reflected border policy. Frame-pair statistics use the median $m=\mathrm{median}(e)$ and **scaled median absolute deviation** $D=1.4826\,\mathrm{median}(|e-m|)$, excluding nonfinite values. The absolute tolerance is

$$\tau=m+k D,\qquad k=\texttt{HardThresholdK}.$$

A sampled candidate passes the *hybrid* mask when

$$
e(\mathbf x)^2\le\alpha\left(\|\mathbf f_{12}(\mathbf x)\|^2+\|\mathbf f_{21}(\mathbf x+\mathbf f_{12})\|^2\right)+\tau^2,
$$

where $\alpha=\texttt{RelativeConsistencyAlpha}$. `ConsistencyErrorThreshold` is a fallback when robust statistics are unavailable, **not always an independent absolute cutoff**. Error and mask are calculated on the dense field and later sampled with `ArrowDensityStep` (offset by step//2). Hard mode exports only consistent sampled vectors. A `median_of_means` scalar is calculated for QC but is not used by the threshold test.

**Functions:** `consistency.compute_flow_pair`, `consistency.median_of_means`.

## 6. Soft confidence mode (optional)

Compute $c(\mathbf x)=\exp[-(e(\mathbf x)/\sigma(\mathbf x))^2]$, where the global scale is $\sigma=m+\texttt{ConfidenceSigma}\,D$. If `UseLocalConfidenceSigma=true`, replace it by a local median plus scaled local MAD over `LocalSigmaKernelSize`; sigma is bounded below by `MinimumConfidenceSigma`.

In soft mode, confidence is multiplied by the hard consistency mask **before** neighborhood smoothing. For a box window $\mathcal N$,

$$
\tilde u(\mathbf x)=\frac{\sum_{\mathcal N(\mathbf x)}c\,u}{\sum_{\mathcal N(\mathbf x)}c+\epsilon},\qquad
\tilde v(\mathbf x)=\frac{\sum_{\mathcal N(\mathbf x)}c\,v}{\sum_{\mathcal N(\mathbf x)}c+\epsilon}.
$$

The final sample must **both** pass the hybrid consistency mask **and** have local confidence $c\ge\texttt{MinConfidence}$. In hard mode, confidence is a diagnostic and is *not* a selection threshold. `VelocityInversion` multiplies output vectors by -1, 0, or +1; for normal forward-flow validation keep +1.

**Functions:** `consistency.calculate_local_sigma`, `consistency.confidence_weighted_smoothing`, `consistency.compute_flow_pair`.

## 7. Processing and outputs

The pipeline processes one adjacent frame pair at a time, optionally with bounded worker concurrency. Numerical fields are saved to disk after each completed pair; QC scalar records are retained. A later disk-backed pass composes velocity video using a common speed scale. The input frame sequence and, when enabled, the temporal SVD matrix remain memory-resident.

**Functions:** `pipeline.run`, `pipeline._process_pairs`, `outputs.save_frame_field`, `outputs.save_validation`, `videos.render_velocity_video`.
