"""Manual Farnebäck pyramid, polynomial fitting, and flow estimation.

The numerical routines and explanatory comments are extracted from the original
research script; configuration is passed explicitly rather than read globally.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Optional
import cv2
import numpy as np
from .config import _UNSET


def build_farneback_visualization_pyramid(
        frame_uint8: np.ndarray,
        pyr_scale: Optional[float] = None,
        levels: Optional[int] = None,
        min_size: Optional[int] = None,
        *, cfg):
    """
    Build the image pyramid used by the explicit/manual Farneback solver.

    The same function is also used by save_pyramid_visualizations(), so the
    diagnostic contact sheets show the exact image levels passed to the manual
    coarse-to-fine calculation.

    Each level is generated directly from level 0 after scale-dependent
    Gaussian pre-smoothing and resizing. This keeps the old diagnostic behavior
    while making it part of the actual solver instead of a visualization-only
    approximation.

    Returns
    -------
    levels : list[dict]
        Each dictionary contains:
          level, scale, width, height, sigma, smooth_kernel, image
    """

    frame_uint8 = np.asarray(frame_uint8)

    if frame_uint8.ndim != 2:
        raise ValueError(
            "Farneback pyramid expects one 2D grayscale frame."
        )

    if frame_uint8.dtype != np.uint8:
        frame_uint8 = np.clip(frame_uint8, 0, 255).astype(np.uint8)

    if pyr_scale is None:
        pyr_scale = float(cfg.PyramidDownsampleFactor)
    else:
        pyr_scale = float(pyr_scale)

    if levels is None:
        requested_levels = max(1, int(cfg.PyramidLevels))
    else:
        requested_levels = max(1, int(levels))

    if min_size is None:
        min_size = max(1, int(cfg.FarnebackPyramidMinSize))
    else:
        min_size = max(1, int(min_size))

    if not (0.0 < pyr_scale < 1.0):
        raise ValueError(
            "PyramidDownsampleFactor must be between 0 and 1."
        )
    
    frame_float = frame_uint8.astype(np.float32)
    
    h0, w0 = frame_float.shape
    level_records = []

    for level in range(requested_levels):
        scale = pyr_scale ** level
        width = max(1, int(round(w0 * scale)))
        height = max(1, int(round(h0 * scale)))

        # Level 0 is always valid. For coarser levels, stop before the image
        # becomes too small for a meaningful local polynomial expansion.
        if level > 0 and min(width, height) < min_size:
            break

        if level == 0:
            sigma = 0.0
            smooth_kernel = 1
            level_image = frame_float.copy()
        else:
            # Scale-dependent pre-smoothing before reduction.
            sigma = (1.0 / scale - 1.0) * 0.5
            smooth_kernel = max(1, int(round(sigma * 5.0)) | 1)

            if smooth_kernel > 1 and sigma > 0.0:
                smoothed = cv2.GaussianBlur(
                    frame_float,
                    (smooth_kernel, smooth_kernel),
                    sigmaX=sigma,
                    sigmaY=sigma,
                    borderType=cv2.BORDER_REFLECT_101,
                )
            else:
                smoothed = frame_float

            level_image = cv2.resize(
                smoothed,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        level_records.append({
            "level": level,
            "scale": float(scale),
            "width": int(width),
            "height": int(height),
            "sigma": float(sigma),
            "smooth_kernel": int(smooth_kernel),
            "image": level_image,
        })

    return level_records


def make_quadratic_fit_kernels(poly_n: int, poly_sigma: float) -> np.ndarray:
    """
    Precompute weighted least-squares kernels for Farneback's local quadratic:

        I(x, y) ~= c + px*x + py*y + qxx*x^2 + qxy*x*y + qyy*y^2

    In matrix notation:

        I(x) ~= x^T A x + b^T x + c

    with
        A = [[qxx, qxy/2],
             [qxy/2, qyy]]
        b = [px, py]^T.

    The returned array has six correlation kernels in coefficient order:
        c, px, py, qxx, qxy, qyy.
    """
    poly_n = int(poly_n)
    poly_sigma = float(poly_sigma)

    if poly_n < 3:
        raise ValueError("PolyFitKernelSize must be at least 3.")
    if poly_n % 2 == 0:
        raise ValueError("PolyFitKernelSize must be odd for the explicit fit.")
    if not np.isfinite(poly_sigma) or poly_sigma <= 0:
        raise ValueError("GaussKernelStdev must be finite and > 0.")

    radius = poly_n // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]

    x = xx.ravel().astype(np.float64)
    y = yy.ravel().astype(np.float64)

    design = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x**2,
        x*y,
        y**2,
    ])

    applicability = np.exp(
        -(x**2 + y**2) / (2.0 * poly_sigma**2)
    )

    xtw = design.T * applicability[None, :]
    normal_matrix = xtw @ design
    fit_operator = np.linalg.pinv(normal_matrix) @ xtw

    kernels = np.stack([
        fit_operator[i].reshape(poly_n, poly_n)
        for i in range(6)
    ], axis=0).astype(np.float32)

    # Worker threads may share this cached array. Make accidental mutation fail.
    kernels.setflags(write=False)
    return kernels


def quadratic_polynomial_expansion(
        image: np.ndarray,
        kernels: np.ndarray,
) -> np.ndarray:
    """Fit the six local quadratic coefficient maps at every pixel."""
    image = image.astype(np.float32, copy=False)

    coefficient_maps = [
        cv2.filter2D(
            image,
            ddepth=cv2.CV_32F,
            kernel=kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        for kernel in kernels
    ]

    return np.stack(coefficient_maps, axis=-1)


def resize_flow_field(
        flow: np.ndarray,
        new_h: int,
        new_w: int,
) -> np.ndarray:
    """Resize a dense flow field and rescale vector lengths to the new grid."""
    old_h, old_w = flow.shape[:2]

    resized = cv2.resize(
        flow,
        (int(new_w), int(new_h)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    resized[..., 0] *= float(new_w) / float(old_w)
    resized[..., 1] *= float(new_h) / float(old_h)
    return resized


def warp_second_frame(
        image2: np.ndarray,
        flow: np.ndarray,
) -> np.ndarray:
    """
    Warp frame 2 toward frame 1 using the current forward-flow estimate.

    Forward convention used by the rest of this pipeline:
        frame1(x) ~= frame2(x + flow(x)).
    """
    h, w = image2.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    return cv2.remap(
        image2,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def sample_farneback_coefficients(
        coeff2: np.ndarray,
        flow: np.ndarray,
        use_opencv_borders: bool = True,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sample the *precomputed* six polynomial maps of Frame 2.

    For a forward flow d from Frame 1 -> Frame 2, the Frame 2 expansion is
    interpolated at (x + d_x, y + d_y). The returned validity mask follows
    OpenCV's native bilinear rule: both a source pixel and its +1 neighbours
    must exist (therefore x < width-1, y < height-1).

    The optional edge weights emulate OpenCV's 5-pixel border attenuation.
    This is OpenCV-*style* coefficient sampling, not a byte-identical port of
    the C++ solver (in particular, cv2.remap quantizes interpolation weights).
    """
    if coeff2.ndim != 3 or coeff2.shape[-1] != 6:
        raise ValueError("Expected coefficient fields with shape (H, W, 6).")
    height, width = coeff2.shape[:2]
    if flow.shape != (height, width, 2):
        raise ValueError("Flow shape does not match polynomial coefficient maps.")

    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    sample_x = xx + flow[..., 0]
    sample_y = yy + flow[..., 1]
    # Use separate masks so the sampler doesn't inspect invalid coordinates.
    valid = (
        np.isfinite(sample_x) & np.isfinite(sample_y)
        & (sample_x >= 0.0) & (sample_y >= 0.0)
        & (sample_x < width - 1) & (sample_y < height - 1)
    )
    safe_x = np.where(valid, sample_x, 0.0).astype(np.float32)
    safe_y = np.where(valid, sample_y, 0.0).astype(np.float32)
    sampled = cv2.remap(
        coeff2, safe_x, safe_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if not use_opencv_borders:
        # The plain ablation samples smoothly all the way to image boundaries.
        sampled = cv2.remap(
            coeff2,
            np.where(np.isfinite(sample_x), sample_x, 0).astype(np.float32),
            np.where(np.isfinite(sample_y), sample_y, 0).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return sampled, np.ones((height, width), dtype=bool), None

    # OpenCV's FarnebackUpdateMatrices zeroes the sampled slopes outside the
    # valid bilinear region and instead uses Frame 1's own quadratic curvature.
    sampled[~valid] = 0.0

    # Native OpenCV's 5-pixel taper: [0.14, 0.14, 0.4472, 0.4472, 0.4472].
    # Applied to both A and Delta-b *before* normal-equation integration.
    border = np.array([0.14, 0.14, 0.4472, 0.4472, 0.4472], dtype=np.float32)
    edge_x = np.minimum(np.arange(width), np.arange(width)[::-1])
    edge_y = np.minimum(np.arange(height), np.arange(height)[::-1])
    weight_x = np.where(edge_x < 5, border[np.minimum(edge_x, 4)], 1.0)
    weight_y = np.where(edge_y < 5, border[np.minimum(edge_y, 4)], 1.0)
    edge_weight = (weight_y[:, None] * weight_x[None, :]).astype(np.float32)
    return sampled, valid, edge_weight


def _integrate_farneback_motion_term(
        field: np.ndarray,
        winsize: int,
        use_gaussian_window: bool,
) -> np.ndarray:
    """Pool one normal-equation term over the Farneback neighborhood."""
    winsize = max(1, int(winsize))

    if use_gaussian_window:
        # OpenCV GaussianBlur requires odd kernel dimensions. The common
        # normalization factor cancels because all normal-equation terms are
        # filtered with the same kernel.
        if winsize % 2 == 0:
            winsize += 1
        return cv2.GaussianBlur(
            field,
            (winsize, winsize),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_REFLECT_101,
        )

    return cv2.boxFilter(
        field,
        ddepth=-1,
        ksize=(winsize, winsize),
        normalize=False,
        borderType=cv2.BORDER_REFLECT_101,
    )


def farneback_residual_increment(
        coeff1: np.ndarray,
        coeff2: np.ndarray,
        winsize: int,
        use_gaussian_window: bool,
        regularization: float,
        max_increment: Optional[float],
        current_flow: Optional[np.ndarray] = None,
        return_information: bool = False,
        solve_absolute: bool = False,
        sampled_valid: Optional[np.ndarray] = None,
        boundary_weight: Optional[np.ndarray] = None,
        *, cfg) -> np.ndarray:
    """
    Solve one explicit Farneback update: original residual or absolute-flow.

    After frame 2 has been warped with the current flow estimate, the remaining
    local displacement delta satisfies approximately

        A delta = Delta b,

    where
        A       = (A1 + A2) / 2
        Delta b = (b1 - b2) / 2.

    Neighboring equations are pooled into the normal system

        G = sum(A^T A),    h = sum(A^T Delta b),    G delta = h.

    In coefficient-warping mode, coeff2 is interpolated from the original
    Frame 2 polynomial maps. Because these were NOT refitted on a warped image,
    Delta b must include A * current_flow (before spatial integration), and
    the updated *absolute* displacement is solved. We return an increment
    (absolute displacement - current_flow) to preserve the existing interface.
    """
    if solve_absolute and current_flow is None:
        raise ValueError("Coefficient-warping requires current_flow.")
    # Midpoint quadratic coefficients. The xy coefficient in the scalar
    # polynomial is 2*Axy, hence the 0.5 factor below.
    a11 = 0.5 * (coeff1[..., 3] + coeff2[..., 3])
    a12 = 0.25 * (coeff1[..., 4] + coeff2[..., 4])
    a22 = 0.5 * (coeff1[..., 5] + coeff2[..., 5])

    delta_b1 = 0.5 * (coeff1[..., 1] - coeff2[..., 1])
    delta_b2 = 0.5 * (coeff1[..., 2] - coeff2[..., 2])

    if solve_absolute:
        if sampled_valid is not None:
            # Match OpenCV's fallback at invalid displaced locations:
            # use Frame 1 curvature and zero Frame 2 linear terms.
            a11 = np.where(sampled_valid, a11, coeff1[..., 3])
            a12 = np.where(sampled_valid, a12, 0.5 * coeff1[..., 4])
            a22 = np.where(sampled_valid, a22, coeff1[..., 5])
            delta_b1 = np.where(sampled_valid, delta_b1, 0.5 * coeff1[..., 1])
            delta_b2 = np.where(sampled_valid, delta_b2, 0.5 * coeff1[..., 2])

        # This is the critical OpenCV coefficient-warping correction.
        # The unknown is the *absolute* displacement; without A*d the
        # previous estimate would be effectively lost/double-counted.
        u0, v0 = current_flow[..., 0], current_flow[..., 1]
        delta_b1 = delta_b1 + a11*u0 + a12*v0
        delta_b2 = delta_b2 + a12*u0 + a22*v0

    if boundary_weight is not None:
        a11 = a11 * boundary_weight
        a12 = a12 * boundary_weight
        a22 = a22 * boundary_weight
        delta_b1 = delta_b1 * boundary_weight
        delta_b2 = delta_b2 * boundary_weight

    # Per-pixel contributions to A^T A and A^T Delta b.
    g11 = a11 * a11 + a12 * a12
    g12 = a11 * a12 + a12 * a22
    g22 = a12 * a12 + a22 * a22

    h1 = a11 * delta_b1 + a12 * delta_b2
    h2 = a12 * delta_b1 + a22 * delta_b2

    # Farneback neighborhood integration.
    g11 = _integrate_farneback_motion_term(g11, winsize, use_gaussian_window)
    g12 = _integrate_farneback_motion_term(g12, winsize, use_gaussian_window)
    g22 = _integrate_farneback_motion_term(g22, winsize, use_gaussian_window)
    h1 = _integrate_farneback_motion_term(h1, winsize, use_gaussian_window)
    h2 = _integrate_farneback_motion_term(h2, winsize, use_gaussian_window)

    # The matrix used for observability must be the RAW normal matrix,
    # before diagonal regularization alters its eigenvalues.
    trace_raw = g11 + g22
    discr = np.sqrt(np.maximum(0.0, (g11 - g22)**2 + 4.0*g12*g12))
    eig_max = np.maximum(0.0, 0.5 * (trace_raw + discr))
    eig_min = np.maximum(0.0, 0.5 * (trace_raw - discr))

    # The original method remains bit-for-bit equivalent in the OFF branch.
    if cfg.UseInformationAwareSolver:
        finite = eig_max[np.isfinite(eig_max)]
        reference = float(np.percentile(finite, 75)) if finite.size else 0.0
        # Avoid arbitrary absolute eigenvalue thresholds for normalized images.
        ridge = max(1e-12, cfg.InformationRidgeFraction * reference)
        rhs1, rhs2 = h1.copy(), h2.copy()

        if cfg.UseInformationSpatialPrior and current_flow is not None:
            # 0..1 local trust: large lambda_min = 2D observable, small = weak.
            trust = eig_min / (eig_min + ridge)
            trust = np.where(np.isfinite(trust), trust, 0).astype(np.float32)
            k = max(3, int(cfg.InformationPriorWindow) | 1)
            # The same box-supported neighborhood supplies a motion prior.
            # In empty neighborhoods the current estimate is the prior, not 0.
            sw = cv2.boxFilter(trust, -1, (k, k), normalize=False,
                               borderType=cv2.BORDER_REFLECT_101)
            mean_u = cv2.boxFilter(trust * current_flow[..., 0], -1, (k, k),
                                   normalize=False, borderType=cv2.BORDER_REFLECT_101)
            mean_v = cv2.boxFilter(trust * current_flow[..., 1], -1, (k, k),
                                   normalize=False, borderType=cv2.BORDER_REFLECT_101)
            prior_u = np.where(sw > 1e-8, mean_u / (sw + 1e-12), current_flow[..., 0])
            prior_v = np.where(sw > 1e-8, mean_v / (sw + 1e-12), current_flow[..., 1])
            if solve_absolute:
                rhs1 += ridge * prior_u
                rhs2 += ridge * prior_v
            else:
                rhs1 += ridge * (prior_u - current_flow[..., 0])
                rhs2 += ridge * (prior_v - current_flow[..., 1])
        elif solve_absolute:
            # If spatial borrowing is disabled, regularize only the *change*.
            rhs1 += ridge * current_flow[..., 0]
            rhs2 += ridge * current_flow[..., 1]
        # Direction-dependent damping in G's eigenbasis, implemented via an
        # equivalent 2x2 ridge solve; small eigendirections are suppressed.
        g11 = g11 + ridge
        g22 = g22 + ridge
    else:
        ridge = np.float32(max(0.0, float(regularization))) * (trace_raw + 1.0)
        rhs1, rhs2 = h1, h2
        if solve_absolute:
            # Plain regularization is centred on the previous estimate.
            rhs1 = rhs1 + ridge * current_flow[..., 0]
            rhs2 = rhs2 + ridge * current_flow[..., 1]
        g11 = g11 + ridge
        g22 = g22 + ridge

    determinant = g11 * g22 - g12 * g12
    du = np.zeros_like(g11, dtype=np.float32)
    dv = np.zeros_like(g11, dtype=np.float32)
    valid = (
        np.isfinite(determinant) & np.isfinite(rhs1) & np.isfinite(rhs2)
        & (np.abs(determinant) > 1e-12)
    )
    du[valid] = (g22[valid]*rhs1[valid] - g12[valid]*rhs2[valid])/determinant[valid]
    dv[valid] = (g11[valid]*rhs2[valid] - g12[valid]*rhs1[valid])/determinant[valid]

    if solve_absolute:
        # On singular pixels keep the previous estimate, not an artificial zero.
        du[~valid] = current_flow[..., 0][~valid]
        dv[~valid] = current_flow[..., 1][~valid]
        # Convert the absolute solution back to a correction for flow += delta.
        du -= current_flow[..., 0]
        dv -= current_flow[..., 1]

    # One pathological local system should not destroy the current coarse-to-
    # fine estimate. This cap applies to a single residual refinement only.
    if max_increment is not None and float(max_increment) > 0:
        magnitude = np.hypot(du, dv)
        too_large = magnitude > float(max_increment)

        scale = np.ones_like(magnitude, dtype=np.float32)
        scale[too_large] = (
            float(max_increment)
            / (magnitude[too_large] + 1e-12)
        )
        du *= scale
        dv *= scale

    delta = np.stack([du, dv], axis=-1).astype(np.float32)
    if not return_information:
        return delta
    # Diagnostic-only arrays. Normal production runs return just delta.
    eps = 1e-30
    return delta, {
        "eig_min_raw": eig_min.astype(np.float32),
        "eig_max_raw": eig_max.astype(np.float32),
        "condition_raw": (eig_max / np.maximum(eig_min, eps)).astype(np.float32),
        "observability": (eig_min / (eig_min + max(float(np.max(np.atleast_1d(ridge))), eps))).astype(np.float32),
        "local_ridge": np.asarray(ridge, dtype=np.float32),
    }


def calc_optical_flow_farneback_explicit(
        image1: np.ndarray,
        image2: np.ndarray,
        pyr_scale: float,
        levels: int,
        winsize: int,
        iterations: int,
        poly_n: int,
        poly_sigma: float,
        flags: int = 0,
        regularization: float = _UNSET,
        max_increment: Optional[float] = _UNSET,
        level_callback=None,
        information_callback=None,
        warp_mode: Optional[str] = None,
        *, cfg) -> np.ndarray:
    """
    Explicit/manual 2D quadratic Farneback optical flow.

    This function intentionally does NOT call cv2.calcOpticalFlowFarneback.
    It implements the main algorithmic steps directly:

      1. construct a coarse-to-fine Gaussian pyramid,
      2. fit a local quadratic polynomial at every pixel,
      3. warp/sample frame 2 at x + current_flow,
      4. estimate the residual displacement from the coefficient change,
      5. pool the local equations over WindowSize and solve G d = h,
      6. iterate at each level and upsample the flow to the next finer level.

    Returns
    -------
    flow : ndarray, shape (H, W, 2), float32
        flow[..., 0] = horizontal displacement u [px/frame]
        flow[..., 1] = vertical displacement v [px/frame]
    """
    if regularization is _UNSET:
        regularization = cfg.ExplicitFarnebackRegularization
    if max_increment is _UNSET:
        max_increment = cfg.ExplicitMaxResidualIncrement
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    if image1.ndim != 2 or image2.ndim != 2:
        raise ValueError("Explicit Farneback expects two 2D grayscale images.")
    if image1.shape != image2.shape:
        raise ValueError("Explicit Farneback input frames must have equal shape.")

    iterations = max(1, int(iterations))
    levels = max(1, int(levels))
    winsize = max(1, int(winsize))

    gaussian_flag = int(cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    unsupported_flags = int(flags) & ~gaussian_flag
    if unsupported_flags:
        raise ValueError(
            "The explicit Farneback solver only supports FilterFlags=0 or "
            "cv2.OPTFLOW_FARNEBACK_GAUSSIAN. "
            f"Unsupported bits: {unsupported_flags}."
        )
    use_gaussian_window = bool(int(flags) & gaussian_flag)
    if warp_mode is None:
        warp_mode = cfg.FarnebackWarpMode
    warp_mode = str(warp_mode).strip().lower()
    if warp_mode not in ("image", "coefficients"):
        raise ValueError("warp_mode must be 'image' or 'coefficients'.")

    pyramid1_records = build_farneback_visualization_pyramid(
        image1,
        pyr_scale=pyr_scale,
        levels=levels,
        min_size=cfg.FarnebackPyramidMinSize,
        cfg=cfg,
    )
    pyramid2_records = build_farneback_visualization_pyramid(
        image2,
        pyr_scale=pyr_scale,
        levels=levels,
        min_size=cfg.FarnebackPyramidMinSize,
        cfg=cfg,
    )

    n_levels = min(len(pyramid1_records), len(pyramid2_records))
    if n_levels == 0:
        raise RuntimeError("Explicit Farneback pyramid contains no valid levels.")

    # The local polynomial fit is intensity-scale invariant in the displacement
    # solve, but [0,1] keeps its numerical values well behaved.
    pyramid1 = [
        rec["image"].astype(np.float32) / 255.0
        for rec in pyramid1_records[:n_levels]
    ]
    pyramid2 = [
        rec["image"].astype(np.float32) / 255.0
        for rec in pyramid2_records[:n_levels]
    ]

    kernels = make_quadratic_fit_kernels(
        int(poly_n),
        float(poly_sigma),
    )

    flow = None

    # Pyramid lists are fine -> coarse, while the solver runs coarse -> fine.
    for level_index in range(n_levels - 1, -1, -1):
        frame1_level = pyramid1[level_index]
        frame2_level = pyramid2[level_index]
        level_h, level_w = frame1_level.shape

        if flow is None:
            flow = np.zeros((level_h, level_w, 2), dtype=np.float32)
        else:
            flow = resize_flow_field(flow, level_h, level_w)

        coeff1 = quadratic_polynomial_expansion(frame1_level, kernels)
        if warp_mode == "coefficients":
            # OpenCV-style: polynomial expansion is performed once for Frame 2.
            coeff2_original = quadratic_polynomial_expansion(frame2_level, kernels)

        for iteration in range(iterations):
            if warp_mode == "coefficients":
                coeff2, sampled_valid, boundary_weight = sample_farneback_coefficients(
                    coeff2_original, flow,
                    use_opencv_borders=cfg.CoefficientWarpOpenCVBorders,
                )
            else:
                # Legacy path preserved: warp image and refit each iteration.
                warped2 = warp_second_frame(frame2_level, flow)
                coeff2 = quadratic_polynomial_expansion(warped2, kernels)
                sampled_valid, boundary_weight = None, None

            want_info = information_callback is not None and iteration == iterations - 1
            output = farneback_residual_increment(
                coeff1=coeff1, coeff2=coeff2, winsize=winsize,
                use_gaussian_window=use_gaussian_window,
                regularization=regularization, max_increment=max_increment,
                current_flow=flow, return_information=want_info,
                solve_absolute=(warp_mode == "coefficients"),
                sampled_valid=sampled_valid, boundary_weight=boundary_weight,
                cfg=cfg,
            )
            if want_info:
                delta_flow, info = output
                information_callback(level_index, pyramid1_records[level_index], info)
            else:
                delta_flow = output
            flow += delta_flow

        # On-demand snapshot after all iterations at the ACTUAL pyramid level.
        # A synchronous callback may save/plot it before flow is resized at the
        # next level. By default there is no callback or additional memory.
        if level_callback is not None:
            level_callback(level_index, pyramid1_records[level_index], flow)

    return flow.astype(np.float32, copy=False)

