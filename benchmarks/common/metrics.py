"""Evaluation metrics shared by all benchmark scenarios.

All displacements are in px per frame pair, image coordinates
(+u right, +v down). Inputs are arrays of equal shape sampled at the
same positions; truth and estimate must already be matched by coordinate.
"""
import numpy as np


def endpoint_error(u, v, u_truth, v_truth):
    """Euclidean endpoint error |d - d_truth| at each sample."""
    return np.hypot(u - u_truth, v - v_truth)


def angular_error_deg(u, v, u_truth, v_truth):
    """Angle between estimated and true displacement in degrees.

    Undefined (NaN) where either vector has zero length, so zero-speed
    ground truth must be evaluated with endpoint error instead.
    """
    est_mag = np.hypot(u, v)
    gt_mag = np.hypot(u_truth, v_truth)
    denom = est_mag * gt_mag

    out = np.full_like(est_mag, np.nan, dtype=np.float64)
    usable = denom > np.finfo(np.float64).eps

    cosang = np.empty_like(est_mag, dtype=np.float64)
    cosang[usable] = (
        u[usable] * u_truth[usable]
        + v[usable] * v_truth[usable]
    ) / denom[usable]
    cosang[usable] = np.clip(cosang[usable], -1.0, 1.0)
    out[usable] = np.degrees(np.arccos(cosang[usable]))
    return out


def safe_median(x):
    """Median of the finite values; NaN if there are none."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def safe_mean(x):
    """Mean of the finite values; NaN if there are none."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def safe_percentile(x, q):
    """q-th percentile of the finite values; NaN if there are none."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")
