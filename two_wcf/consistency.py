"""Forward/backward consistency, confidence, sampling, and QC scalars."""
from __future__ import annotations
from typing import Optional
import numpy as np
import cv2
from scipy.ndimage import median_filter
from .config import _UNSET
from .image_io import normalize_frame
from .solver import calc_optical_flow_farneback_explicit


def median_of_means(x: np.ndarray, group_size: int = _UNSET,        *, cfg) -> float:
    """
    Robust statistic: split x into groups of size `group_size`,
    compute mean in each group, then return the median of those means.
    Very robust to outliers for large arrays.
    """
    if group_size is _UNSET:
        group_size = cfg.MedianOfMeansGroupSize
    x = np.asarray(x)
    # Keep only finite values
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    # A small image may have fewer valid pixels than one requested group.
    # In that case the only group is the whole array, whose mean is the MOM.
    if n < group_size:
        return float(np.mean(x))
    g = max(1, n // group_size)
    trimmed = x[:g * group_size]
    if trimmed.size == 0:
        return float("nan")
    groups = trimmed.reshape(g, group_size)
    group_means = groups.mean(axis=1)
    return float(np.median(group_means))


def calculate_local_sigma(consistency_error: np.ndarray,        *, cfg) -> tuple:
    """
    Calculate spatially varying local error statistics.

    Returns
    -------
    sigma_map : np.ndarray
        Local confidence scale:

            sigma_local(x, y)
                = local_median_error
                + ConfidenceSigma * local_scaled_MAD

    local_mad : np.ndarray
        Local scaled median absolute deviation:

            local_MAD
                = 1.4826 * median(|error - local_median|)
    """

    error = consistency_error.astype(np.float32)
    
    # Size of the local neighbourhood.
    # With LocalSigmaKernelSize = 21, every pixel looks at a
    # 21 x 21 neighbourhood around itself.
    k = cfg.LocalSigmaKernelSize

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    # Local median consistency error
    local_median = median_filter(
        error,
        size = k,
        mode  = "nearest"
    ).astype(np.float32)

    # Absolute deviation from the local median
    #For every pixel: deviation = |error - local_median|
    local_abs_deviation = np.abs(
        error - local_median
    ).astype(np.float32)

    # Local median absolute deviation
    # The factor 1.4826 scales MAD so that, for approximately Gaussian data, it behaves similarly to a standard deviation 
    local_mad = 1.4826 * median_filter(
        local_abs_deviation,
        size = k,
        mode = "nearest"
    )
    
    #Sigma_local = Local median error + ConfidenceSigma * Local MAD
    sigma_map = (
        local_median
        + cfg.ConfidenceSigma * local_mad
    ).astype(np.float32)

    # Prevent zero or extremely small sigma because it is later used as a divisor
    sigma_map = np.maximum(
        sigma_map,
        cfg.MinimumConfidenceSigma
    ).astype(np.float32)

    return sigma_map, local_mad


def confidence_weighted_smoothing(
        u: np.ndarray,
        v: np.ndarray,
        consistency_error: np.ndarray,
        sigma: float,
        reliability_mask: Optional[np.ndarray] = None
,        *, cfg) -> tuple:
    """
    Compute confidence map from consistency error and apply
    confidence-weighted local smoothing to the flow.

    c(x,y) = exp( - (err / σ)^2 )

    Smoothing:
        u_s = boxfilter(c * u) / boxfilter(c)
        v_s = boxfilter(c * v) / boxfilter(c)

    Parameters
    ----------
    u, v : np.ndarray
        Raw flow components.
    consistency_error : np.ndarray
        L2 forward-backward consistency error at each pixel.
    sigma : float
        Scale parameter for the confidence decay. Should be > 0.
        In this pipeline it is typically derived per frame pair as
        σ_frame = median_err + ConfidenceSigma * MAD.

    Returns
    -------
    u_smoothed, v_smoothed, confidence_map
    """
    sigma_safe = np.asarray(sigma,dtype=np.float32)
    
    sigma_safe = np.where(
    np.isfinite(sigma_safe)
    & (sigma_safe > cfg.MinimumConfidenceSigma),
    sigma_safe,
    cfg.MinimumConfidenceSigma)

   # Confidence in [0, 1]
    conf = np.exp(-(consistency_error / sigma_safe) ** 2).astype(np.float32)
    
    # Completely prevent inconsistent vectors from contributing
    # to the confidence-weighted neighbourhood average
    if reliability_mask is not None:
        conf *= reliability_mask.astype(np.float32)
    
    k = cfg.ConfidenceKernelSize
    if k < 1:
        k = 1
    
    if k % 2 == 0:
        k += 1
    
    weight_sum = cv2.boxFilter(
        conf,
        ddepth=-1,
        ksize=(k, k),
        normalize=False
    )
    
    u_weighted = cv2.boxFilter(
        u * conf,
        ddepth=-1,
        ksize=(k, k),
        normalize=False
    )
    
    v_weighted = cv2.boxFilter(
        v * conf,
        ddepth=-1,
        ksize=(k, k),
        normalize=False
    )
    
    eps = 1e-6
    u_s = u_weighted / (weight_sum + eps)
    v_s = v_weighted / (weight_sum + eps)
    
    return (
        u_s.astype(np.float32),
        v_s.astype(np.float32),
        conf
    )


def compute_flow_pair(index: int, images, h: int, w: int, global_min, global_max,        *, cfg):
    """
    Compute two-way-consistent explicit/manual Farneback flow for frame pair (index, index+1).

    MEMORY DESIGN
    -------------
    Only data that must survive this function are returned:
      * sampled/final vectors,
      * scalar QC diagnostics,
      * sampled validation arrays,
      * full-resolution diagnostic maps ONLY when SaveDiagnosticMaps=True.

    Large local-MAD/local-sigma maps are not calculated at all unless they are
    actually needed by local-confidence processing or diagnostic-map output.
    This prevents every completed Future from carrying unnecessary HxW arrays.
    """
    prev_img = normalize_frame(images[index], global_min, global_max)
    next_img = normalize_frame(images[index + 1], global_min, global_max)

    # Forward explicit/manual Farneback flow
    flow_fwd = calc_optical_flow_farneback_explicit(
        prev_img,
        next_img,
        pyr_scale=cfg.PyramidDownsampleFactor,
        levels=cfg.PyramidLevels,
        winsize=cfg.WindowSize,
        iterations=cfg.IterationCount,
        poly_n=cfg.PolyFitKernelSize,
        poly_sigma=cfg.GaussKernelStdev,
        flags=cfg.FilterFlags,
        regularization=cfg.ExplicitFarnebackRegularization,
        max_increment=cfg.ExplicitMaxResidualIncrement,
        cfg=cfg,
    )
    u = flow_fwd[..., 0]
    v = flow_fwd[..., 1]

    # Backward explicit/manual Farneback flow for consistency
    flow_bwd = calc_optical_flow_farneback_explicit(
        next_img,
        prev_img,
        pyr_scale=cfg.PyramidDownsampleFactor,
        levels=cfg.PyramidLevels,
        winsize=cfg.WindowSize,
        iterations=cfg.IterationCount,
        poly_n=cfg.PolyFitKernelSize,
        poly_sigma=cfg.GaussKernelStdev,
        flags=cfg.FilterFlags,
        regularization=cfg.ExplicitFarnebackRegularization,
        max_increment=cfg.ExplicitMaxResidualIncrement,
        cfg=cfg,
    )

    # float32 coordinate grids use half the memory of the previous platform-
    # default int64 meshgrid and are exactly what cv2.remap needs downstream.
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    coords_x = grid_x + u
    coords_y = grid_y + v

    bwd_u_warped = cv2.remap(
        flow_bwd[..., 0], coords_x, coords_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    bwd_v_warped = cv2.remap(
        flow_bwd[..., 1], coords_x, coords_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # L2 forward-backward consistency error
    consistency_error = np.sqrt(
        (u + bwd_u_warped) ** 2
        + (v + bwd_v_warped) ** 2
    )

    # --- robust error stats (per frame pair) ---
    err_flat = consistency_error.ravel()
    err_flat = err_flat[np.isfinite(err_flat)]

    if err_flat.size > 0:
        med = float(np.median(err_flat))
        mad_raw = float(np.median(np.abs(err_flat - med)))
        mad = 1.4826 * mad_raw
        mom = median_of_means(err_flat, group_size=cfg.MedianOfMeansGroupSize, cfg=cfg)
    else:
        med = float("nan")
        mad = float("nan")
        mom = float("nan")

    if np.isfinite(med) and np.isfinite(mad):
        thr_frame = med + cfg.HardThresholdK * mad
        sigma_frame = med + cfg.ConfidenceSigma * mad
    else:
        # Reached only when no pixel has a finite consistency error. Every
        # vector is then rejected by the mask below, so these values only
        # affect the recorded QC scalars. Both come from the configuration.
        thr_frame = cfg.ConsistencyErrorThreshold
        sigma_frame = cfg.MinimumConfidenceSigma

    if not np.isfinite(thr_frame):
        thr_frame = cfg.ConsistencyErrorThreshold
    if not np.isfinite(sigma_frame) or sigma_frame <= 0:
        sigma_frame = cfg.MinimumConfidenceSigma

    # Sample grid
    sample_y = slice(cfg.ArrowDensityStep // 2, h, cfg.ArrowDensityStep)
    sample_x = slice(cfg.ArrowDensityStep // 2, w, cfg.ArrowDensityStep)

    y_grid, x_grid = np.mgrid[
        cfg.ArrowDensityStep // 2:h:cfg.ArrowDensityStep,
        cfg.ArrowDensityStep // 2:w:cfg.ArrowDensityStep,
    ]

    # -----------------------------------------------------------------
    # Hybrid forward/backward consistency test
    # -----------------------------------------------------------------
    forward_magnitude_sq = u**2 + v**2
    backward_magnitude_sq = bwd_u_warped**2 + bwd_v_warped**2

    allowed_error_squared = (
        cfg.RelativeConsistencyAlpha
        * (forward_magnitude_sq + backward_magnitude_sq)
        + thr_frame**2
    )

    reliability_mask = consistency_error**2 <= allowed_error_squared

    effective_error_limit = np.sqrt(
        np.maximum(allowed_error_squared, 0.0)
    )
    effective_error_limit_finite = effective_error_limit[
        np.isfinite(effective_error_limit)
    ]

    if effective_error_limit_finite.size > 0:
        effective_limit_median = float(np.median(effective_error_limit_finite))
        effective_limit_p10 = float(np.percentile(effective_error_limit_finite, 10))
        effective_limit_p90 = float(np.percentile(effective_error_limit_finite, 90))
    else:
        effective_limit_median = float("nan")
        effective_limit_p10 = float("nan")
        effective_limit_p90 = float("nan")

    # Sample before the full-resolution temporaries go out of scope.
    u_raw_sampled = u[sample_y, sample_x]
    v_raw_sampled = v[sample_y, sample_x]
    consistency_error_sampled = consistency_error[sample_y, sample_x]
    mask_sampled = reliability_mask[sample_y, sample_x]

    # -----------------------------------------------------------------
    # Local statistics are expensive HxW maps.
    # Calculate them ONLY if they are actually needed.
    # -----------------------------------------------------------------
    need_local_stats = cfg.UseLocalConfidenceSigma or cfg.SaveDiagnosticMaps
    local_sigma_map = None
    local_mad_map = None

    if need_local_stats:
        local_sigma_map, local_mad_map = calculate_local_sigma(
            consistency_error,
            cfg=cfg,
        )

    confidence_map_for_diagnostics = None

    if cfg.UseConfidenceSmoothing:
        # Soft mode genuinely needs a dense confidence field because the
        # confidence-weighted box filters operate on every pixel.
        sigma_for_confidence = (
            local_sigma_map if cfg.UseLocalConfidenceSigma else sigma_frame
        )

        u_s, v_s, conf_map = confidence_weighted_smoothing(
            u,
            v,
            consistency_error,
            sigma_for_confidence,
            reliability_mask,
            cfg=cfg,
        )

        u_sampled = u_s[sample_y, sample_x]
        v_sampled = v_s[sample_y, sample_x]
        conf_sampled = conf_map[sample_y, sample_x]

        valid_indices = (
            mask_sampled
            & (conf_sampled >= cfg.MinConfidence)
        )

        if cfg.SaveDiagnosticMaps:
            confidence_map_for_diagnostics = conf_map

    else:
        # Hard mode does not need a dense confidence map unless diagnostic
        # map output was explicitly requested. For validation, calculate
        # confidence only on the sampled grid.
        u_sampled = u_raw_sampled
        v_sampled = v_raw_sampled

        if cfg.UseLocalConfidenceSigma:
            sigma_sampled = local_sigma_map[sample_y, sample_x]
        else:
            sigma_sampled = np.float32(sigma_frame)

        sigma_sampled = np.asarray(sigma_sampled, dtype=np.float32)
        sigma_sampled = np.where(
            np.isfinite(sigma_sampled)
            & (sigma_sampled > cfg.MinimumConfidenceSigma),
            sigma_sampled,
            cfg.MinimumConfidenceSigma,
        )

        conf_sampled = np.exp(
            -(consistency_error_sampled / sigma_sampled) ** 2
        ).astype(np.float32)

        valid_indices = mask_sampled

        if cfg.SaveDiagnosticMaps:
            if cfg.UseLocalConfidenceSigma:
                sigma_full = local_sigma_map
            else:
                sigma_full = np.float32(sigma_frame)

            sigma_full = np.asarray(sigma_full, dtype=np.float32)
            sigma_full = np.where(
                np.isfinite(sigma_full)
                & (sigma_full > cfg.MinimumConfidenceSigma),
                sigma_full,
                cfg.MinimumConfidenceSigma,
            )

            confidence_map_for_diagnostics = np.exp(
                -(consistency_error / sigma_full) ** 2
            ).astype(np.float32)

    rejected_indices = ~valid_indices
    consistency_rejected_indices = ~mask_sampled

    x_rejected = x_grid[rejected_indices]
    y_rejected = y_grid[rejected_indices]

    total_vectors = valid_indices.size
    retained_vectors = int(np.count_nonzero(valid_indices))
    rejected_vectors = int(total_vectors - retained_vectors)
    consistency_rejected_vectors = int(
        np.count_nonzero(consistency_rejected_indices)
    )

    if cfg.UseConfidenceSmoothing:
        low_confidence_vectors = int(np.count_nonzero(
            mask_sampled & (conf_sampled < cfg.MinConfidence)
        ))
    else:
        low_confidence_vectors = 0

    retention_fraction = retained_vectors / max(total_vectors, 1)
    rejection_fraction = rejected_vectors / max(total_vectors, 1)

    print(
        f"[Frame {index}] "
        f"retained {retained_vectors}/{total_vectors} vectors "
        f"({100 * retention_fraction:.1f}%), "
        f"rejected {rejected_vectors}/{total_vectors} "
        f"({100 * rejection_fraction:.1f}%), "
        f"FB-consistency rejected={consistency_rejected_vectors}, "
        f"low-confidence rejected={low_confidence_vectors}, "
        f"thr_frame={thr_frame:.4f} px, "
        f"relative alpha={cfg.RelativeConsistencyAlpha:.4g}, "
        f"effective limit median={effective_limit_median:.4f} px "
        f"(p10={effective_limit_p10:.4f}, p90={effective_limit_p90:.4f}), "
        f"median error={med:.4f}, MAD={mad:.4f}"
    )

    u_output_sampled = cfg.VelocityInversion * u_sampled
    v_output_sampled = cfg.VelocityInversion * v_sampled

    x_valid = x_grid[valid_indices]
    y_valid = y_grid[valid_indices]
    u_valid = u_output_sampled[valid_indices]
    v_valid = v_output_sampled[valid_indices]

    result = {
        "index": index,

        "x": x_valid.astype(float),
        "y": y_valid.astype(float),
        "u": u_valid.astype(float),
        "v": v_valid.astype(float),

        "x_rejected": x_rejected.astype(float),
        "y_rejected": y_rejected.astype(float),

        "median_err": med,
        "mad_err": mad,
        "mom_err": mom,
        "thr_frame": float(thr_frame),
        "sigma_frame": float(sigma_frame),
        "relative_consistency_alpha": float(cfg.RelativeConsistencyAlpha),
        "effective_limit_median": effective_limit_median,
        "effective_limit_p10": effective_limit_p10,
        "effective_limit_p90": effective_limit_p90,
        "total_vectors": int(total_vectors),
        "retained_vectors": retained_vectors,
        "rejected_vectors": rejected_vectors,
        "consistency_rejected_vectors": consistency_rejected_vectors,
        "low_confidence_vectors": low_confidence_vectors,
        "retention_fraction": float(retention_fraction),
        "rejection_fraction": float(rejection_fraction),

        # Explicit copies ensure these small sampled arrays do not retain
        # references to their much larger full-resolution parent arrays.
        "validation_x_all": x_grid.astype(np.float32, copy=True),
        "validation_y_all": y_grid.astype(np.float32, copy=True),
        "validation_u_raw": u_raw_sampled.astype(np.float32, copy=True),
        "validation_v_raw": v_raw_sampled.astype(np.float32, copy=True),
        "validation_u_processed": u_sampled.astype(np.float32, copy=True),
        "validation_v_processed": v_sampled.astype(np.float32, copy=True),
        "validation_u_output": u_output_sampled.astype(np.float32, copy=True),
        "validation_v_output": v_output_sampled.astype(np.float32, copy=True),
        "validation_valid_mask": valid_indices.astype(bool, copy=True),
        "validation_consistency_mask": mask_sampled.astype(bool, copy=True),
        "validation_consistency_error": consistency_error_sampled.astype(
            np.float32, copy=True
        ),
        "validation_confidence": conf_sampled.astype(np.float32, copy=True),
    }

    # Full-resolution maps are returned only when the user asked to save them.
    if cfg.SaveDiagnosticMaps:
        result["local_mad_map"] = local_mad_map.astype(np.float32, copy=False)
        result["local_sigma_map"] = local_sigma_map.astype(np.float32, copy=False)
        result["confidence_map"] = confidence_map_for_diagnostics.astype(
            np.float32, copy=False
        )

    return result

