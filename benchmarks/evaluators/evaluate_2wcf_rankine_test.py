from pathlib import Path
import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# USER SETTINGS
# =============================================================================
# Folder created by generate_rankine_vortex_test.py
SYNTHETIC_TEST_DIR = Path(
    r"E:/Ansis Zivers/Optical Flow/Image_sequences/SYNTHETIC/Rotational_flow/synthetic_rankine_R80_vb3_cx255p5_cy255p5"
)
# Set this to the matching output folder created by the 2WCF pipeline.
# It must contain a subfolder called "validation".
PIPELINE_OUTPUT_DIR = Path(
    r"E:/Ansis Zivers/Optical Flow/Output_2WCF_SVD/SYNTHETIC/Rotational flow/synthetic_rankine_R80_vb3_cx255p5_cy255p5_run_1"
)

# Ignore the outer image border in the primary metrics.
INTERIOR_BORDER_PX = 24

# Relative error is unstable arbitrarily close to the vortex centre because
# true displacement tends to zero there. Only use points above this speed.
MIN_RELATIVE_GT_SPEED_PX = 0.25

# Radial summary resolution.
N_RADIAL_BINS = 14
# =============================================================================


GT_PATH = SYNTHETIC_TEST_DIR / "true_vector_values.npz"
VALIDATION_DIR = PIPELINE_OUTPUT_DIR / "validation"
RESULT_DIR = PIPELINE_OUTPUT_DIR / "synthetic_rankine_validation_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not GT_PATH.exists():
    raise FileNotFoundError(f"True vector file not found: {GT_PATH}")

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder not found: {VALIDATION_DIR}\n"
        "Make sure SaveValidationData = True in the 2WCF script."
    )

validation_files = sorted(VALIDATION_DIR.glob("validation_*.npz"))
if not validation_files:
    raise FileNotFoundError(f"No validation_*.npz files found in {VALIDATION_DIR}")

with np.load(GT_PATH) as gt:
    u_gt_full = gt["u_gt"].astype(np.float64)
    v_gt_full = gt["v_gt"].astype(np.float64)
    speed_gt_full = gt["speed_gt"].astype(np.float64)
    radius_gt_full = gt["radius_gt"].astype(np.float64)
    radial_gt_full = gt["radial_gt"].astype(np.float64)
    tangential_gt_full = gt["tangential_gt"].astype(np.float64)
    continuous_tangential_speed_gt_full = gt["continuous_tangential_speed_gt"].astype(np.float64)
    continuous_vorticity_gt_full = gt["continuous_vorticity_gt"].astype(np.float64)
    continuous_divergence_gt_full = gt["continuous_divergence_gt"].astype(np.float64)
    center_x = float(gt["center_x"])
    center_y = float(gt["center_y"])
    core_radius_px = float(gt["core_radius_px"])
    boundary_tangential_speed_px = float(gt["boundary_tangential_speed_px"])
    omega_core_rad_per_frame = float(gt["omega_core_rad_per_frame"])

HEIGHT, WIDTH = u_gt_full.shape


def endpoint_error(u, v, u_truth, v_truth):
    return np.hypot(u - u_truth, v - v_truth)



def angular_error_deg(u, v, u_truth, v_truth):
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
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")



def safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")



def safe_percentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")



def radial_tangential_components(x, y, u, v):
    dx = x - center_x
    dy = y - center_y
    r = np.hypot(dx, dy)

    er_x = np.full_like(r, np.nan, dtype=np.float64)
    er_y = np.full_like(r, np.nan, dtype=np.float64)
    et_x = np.full_like(r, np.nan, dtype=np.float64)
    et_y = np.full_like(r, np.nan, dtype=np.float64)

    usable = r > np.finfo(np.float64).eps
    er_x[usable] = dx[usable] / r[usable]
    er_y[usable] = dy[usable] / r[usable]

    # Positive tangential direction matches positive image-coordinate rotation.
    et_x[usable] = -dy[usable] / r[usable]
    et_y[usable] = dx[usable] / r[usable]

    radial = u * er_x + v * er_y
    tangential = u * et_x + v * et_y
    return r, radial, tangential



def estimate_center_from_vectors(x, y, u, v, mask):
    """
    Estimate vortex centre as the point c for which the finite displacement is
    orthogonal to the radius vector, i.e. d · (x-c) = 0.

    This yields the linear equation:
        u * c_x + v * c_y = u * x + v * y
    solved in least squares.
    """
    mask = np.asarray(mask, dtype=bool)
    finite = (
        mask
        & np.isfinite(x) & np.isfinite(y)
        & np.isfinite(u) & np.isfinite(v)
    )

    if np.count_nonzero(finite) < 3:
        return float("nan"), float("nan"), float("nan")

    xf = np.asarray(x[finite], dtype=np.float64).ravel()
    yf = np.asarray(y[finite], dtype=np.float64).ravel()
    uf = np.asarray(u[finite], dtype=np.float64).ravel()
    vf = np.asarray(v[finite], dtype=np.float64).ravel()

    A = np.column_stack([uf, vf])
    b = uf * xf + vf * yf

    if np.linalg.matrix_rank(A) < 2:
        return float("nan"), float("nan"), float("nan")

    centre_est, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx_est = float(centre_est[0])
    cy_est = float(centre_est[1])
    c_err = float(np.hypot(cx_est - center_x, cy_est - center_y))
    return cx_est, cy_est, c_err


records = []
spatial_epe_stack = []
reference_x = None
reference_y = None

# Radial aggregation across all frame pairs.
radial_bin_edges = None
radial_epe_values = [[] for _ in range(N_RADIAL_BINS)]
radial_radial_bias_values = [[] for _ in range(N_RADIAL_BINS)]
radial_tangential_bias_values = [[] for _ in range(N_RADIAL_BINS)]
radial_abs_tangential_error_values = [[] for _ in range(N_RADIAL_BINS)]
radial_est_tangential_values = [[] for _ in range(N_RADIAL_BINS)]
radial_true_tangential_values = [[] for _ in range(N_RADIAL_BINS)]
radial_total_counts = np.zeros(N_RADIAL_BINS, dtype=np.int64)
radial_retained_counts = np.zeros(N_RADIAL_BINS, dtype=np.int64)

for pair_index, path in enumerate(validation_files):
    with np.load(path) as data:
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)

        u_raw = data["u_raw"].astype(np.float64)
        v_raw = data["v_raw"].astype(np.float64)
        u_processed = data["u_processed"].astype(np.float64)
        v_processed = data["v_processed"].astype(np.float64)
        u_output = data["u_output"].astype(np.float64)
        v_output = data["v_output"].astype(np.float64)
        valid_mask = data["valid_mask"].astype(bool)
        consistency_mask = data["consistency_mask"].astype(bool)
        consistency_error = data["consistency_error"].astype(np.float64)
        confidence = data["confidence"].astype(np.float64)

    xi = x.astype(np.int64)
    yi = y.astype(np.int64)

    if reference_x is None:
        reference_x = x.copy()
        reference_y = y.copy()
    elif not (np.array_equal(x, reference_x) and np.array_equal(y, reference_y)):
        raise ValueError("Sample grid changed between frame pairs; cannot aggregate spatial maps safely.")

    u_truth = u_gt_full[yi, xi]
    v_truth = v_gt_full[yi, xi]
    speed_truth = speed_gt_full[yi, xi]
    radius_truth = radius_gt_full[yi, xi]
    radial_truth = radial_gt_full[yi, xi]
    tangential_truth = tangential_gt_full[yi, xi]

    interior = (
        (x >= INTERIOR_BORDER_PX) &
        (x < WIDTH - INTERIOR_BORDER_PX) &
        (y >= INTERIOR_BORDER_PX) &
        (y < HEIGHT - INTERIOR_BORDER_PX)
    )

    epe_raw = endpoint_error(u_raw, v_raw, u_truth, v_truth)
    epe_processed = endpoint_error(u_processed, v_processed, u_truth, v_truth)
    epe_output = endpoint_error(u_output, v_output, u_truth, v_truth)
    angle_output = angular_error_deg(u_output, v_output, u_truth, v_truth)

    _, radial_est, tangential_est = radial_tangential_components(x, y, u_output, v_output)

    final = valid_mask
    final_interior = valid_mask & interior
    reliable_relative = final_interior & (speed_truth >= MIN_RELATIVE_GT_SPEED_PX)
    inside_core = final_interior & (radius_truth <= core_radius_px)
    outside_core = final_interior & (radius_truth > core_radius_px)

    total = int(valid_mask.size)
    retained = int(np.count_nonzero(final))
    interior_total = int(np.count_nonzero(interior))
    interior_retained = int(np.count_nonzero(final_interior))

    est_cx, est_cy, center_error = estimate_center_from_vectors(
        x, y, u_output, v_output, final_interior
    )

    record = {
        "frame_pair": pair_index,
        "total_sampled": total,
        "retained": retained,
        "retention_percent": 100.0 * retained / max(total, 1),
        "interior_total": interior_total,
        "interior_retained": interior_retained,
        "interior_retention_percent": 100.0 * interior_retained / max(interior_total, 1),

        "raw_median_epe_px": safe_median(epe_raw[interior]),
        "processed_median_epe_px": safe_median(epe_processed[interior]),

        "final_median_epe_px": safe_median(epe_output[final_interior]),
        "final_mean_epe_px": safe_mean(epe_output[final_interior]),
        "final_p95_epe_px": safe_percentile(epe_output[final_interior], 95),
        "final_median_angle_error_deg": safe_median(angle_output[reliable_relative]),
        "final_median_relative_epe_percent": (
            100.0 * safe_median(epe_output[reliable_relative] / speed_truth[reliable_relative])
            if np.count_nonzero(reliable_relative) else float("nan")
        ),

        "final_median_radial_bias_px": safe_median(radial_est[final_interior] - radial_truth[final_interior]),
        "final_median_tangential_bias_px": safe_median(tangential_est[final_interior] - tangential_truth[final_interior]),
        "final_median_abs_radial_error_px": safe_median(np.abs(radial_est[final_interior] - radial_truth[final_interior])),
        "final_median_abs_tangential_error_px": safe_median(np.abs(tangential_est[final_interior] - tangential_truth[final_interior])),
        "final_median_inside_core_epe_px": safe_median(epe_output[inside_core]),
        "final_median_outside_core_epe_px": safe_median(epe_output[outside_core]),

        "median_fb_consistency_error_px": safe_median(consistency_error[interior]),
        "median_confidence": safe_median(confidence[interior]),
        "consistency_pass_percent": 100.0 * np.count_nonzero(consistency_mask & interior) / max(interior_total, 1),

        "estimated_center_x_px": est_cx,
        "estimated_center_y_px": est_cy,
        "center_error_px": center_error,
    }
    records.append(record)

    epe_for_map = np.where(valid_mask, epe_output, np.nan)
    spatial_epe_stack.append(epe_for_map)

    # Radial aggregation.
    if radial_bin_edges is None:
        max_radius = float(np.max(radius_truth[interior]))
        radial_bin_edges = np.linspace(0.0, max_radius, N_RADIAL_BINS + 1)

    radial_indices = np.digitize(radius_truth, radial_bin_edges) - 1

    for bin_idx in range(N_RADIAL_BINS):
        in_bin = (radial_indices == bin_idx) & interior
        in_bin_retained = in_bin & valid_mask

        radial_total_counts[bin_idx] += int(np.count_nonzero(in_bin))
        radial_retained_counts[bin_idx] += int(np.count_nonzero(in_bin_retained))

        if np.any(in_bin_retained):
            radial_epe_values[bin_idx].extend(epe_output[in_bin_retained].tolist())
            radial_radial_bias_values[bin_idx].extend(
                (radial_est[in_bin_retained] - radial_truth[in_bin_retained]).tolist()
            )
            radial_tangential_bias_values[bin_idx].extend(
                (tangential_est[in_bin_retained] - tangential_truth[in_bin_retained]).tolist()
            )
            radial_abs_tangential_error_values[bin_idx].extend(
                np.abs(tangential_est[in_bin_retained] - tangential_truth[in_bin_retained]).tolist()
            )
            radial_est_tangential_values[bin_idx].extend(tangential_est[in_bin_retained].tolist())
            radial_true_tangential_values[bin_idx].extend(tangential_truth[in_bin_retained].tolist())


# =============================================================================
# SAVE PER-FRAME CSV
# =============================================================================
CSV_PATH = RESULT_DIR / "rankine_validation_per_frame.csv"
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)


# =============================================================================
# AGGREGATED SUMMARY
# =============================================================================
keys = records[0].keys()
summary = {}
for key in keys:
    if key in {"frame_pair", "total_sampled", "retained", "interior_total", "interior_retained"}:
        continue
    values = np.array([r[key] for r in records], dtype=np.float64)
    summary[key] = safe_median(values)

# Radial profile table.
radial_profile_records = []
for bin_idx in range(N_RADIAL_BINS):
    r0 = radial_bin_edges[bin_idx]
    r1 = radial_bin_edges[bin_idx + 1]
    rc = 0.5 * (r0 + r1)
    radial_profile_records.append({
        "bin_index": bin_idx,
        "radius_min_px": r0,
        "radius_max_px": r1,
        "radius_center_px": rc,
        "retention_percent": 100.0 * radial_retained_counts[bin_idx] / max(radial_total_counts[bin_idx], 1),
        "median_epe_px": safe_median(radial_epe_values[bin_idx]),
        "median_radial_bias_px": safe_median(radial_radial_bias_values[bin_idx]),
        "median_tangential_bias_px": safe_median(radial_tangential_bias_values[bin_idx]),
        "median_abs_tangential_error_px": safe_median(radial_abs_tangential_error_values[bin_idx]),
        "median_estimated_tangential_px": safe_median(radial_est_tangential_values[bin_idx]),
        "median_true_tangential_px": safe_median(radial_true_tangential_values[bin_idx]),
    })

RADIAL_CSV_PATH = RESULT_DIR / "rankine_radial_profile.csv"
with open(RADIAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(radial_profile_records[0].keys()))
    writer.writeheader()
    writer.writerows(radial_profile_records)

radius_centers = np.array([r["radius_center_px"] for r in radial_profile_records], dtype=np.float64)
median_true_t = np.array([r["median_true_tangential_px"] for r in radial_profile_records], dtype=np.float64)
median_est_t = np.array([r["median_estimated_tangential_px"] for r in radial_profile_records], dtype=np.float64)
retention_vs_radius = np.array([r["retention_percent"] for r in radial_profile_records], dtype=np.float64)
median_epe_vs_radius = np.array([r["median_epe_px"] for r in radial_profile_records], dtype=np.float64)
median_radial_bias_vs_radius = np.array([r["median_radial_bias_px"] for r in radial_profile_records], dtype=np.float64)
median_tangential_bias_vs_radius = np.array([r["median_tangential_bias_px"] for r in radial_profile_records], dtype=np.float64)
median_abs_t_error_vs_radius = np.array([r["median_abs_tangential_error_px"] for r in radial_profile_records], dtype=np.float64)

# Simple recovered core-radius proxy: radius bin where recovered median tangential
# displacement reaches its maximum.
if np.any(np.isfinite(median_est_t)):
    recovered_core_radius_px = float(radius_centers[np.nanargmax(median_est_t)])
else:
    recovered_core_radius_px = float("nan")

print("\n============================================================")
print("2WCF SYNTHETIC RANKINE-VORTEX VALIDATION")
print("============================================================")
print(f"Vortex centre:             ({center_x:.3f}, {center_y:.3f}) px")
print(f"Core radius:               {core_radius_px:.6f} px")
print(f"Tangential speed at R:     {boundary_tangential_speed_px:.6f} px/frame")
print(f"Core angular speed:        {omega_core_rad_per_frame:.6f} rad/frame")
print(f"Frame pairs analysed:      {len(records)}")
print(f"Interior border:           {INTERIOR_BORDER_PX} px")
print()
print("FINAL PIPELINE OUTPUT — median across frame pairs")
print(f"median endpoint error:           {summary['final_median_epe_px']:.6f} px/frame")
print(f"mean endpoint error:             {summary['final_mean_epe_px']:.6f} px/frame")
print(f"95th-percentile EPE:             {summary['final_p95_epe_px']:.6f} px/frame")
print(f"median relative EPE:             {summary['final_median_relative_epe_percent']:.3f} %")
print(f"median angular error:            {summary['final_median_angle_error_deg']:.4f} deg")
print(f"median radial bias:              {summary['final_median_radial_bias_px']:.6f} px/frame")
print(f"median tangential bias:          {summary['final_median_tangential_bias_px']:.6f} px/frame")
print(f"median |radial error|:           {summary['final_median_abs_radial_error_px']:.6f} px/frame")
print(f"median |tangential error|:       {summary['final_median_abs_tangential_error_px']:.6f} px/frame")
print(f"inside-core median EPE:          {summary['final_median_inside_core_epe_px']:.6f} px/frame")
print(f"outside-core median EPE:         {summary['final_median_outside_core_epe_px']:.6f} px/frame")
print(f"interior retention:              {summary['interior_retention_percent']:.3f} %")
print(f"consistency pass fraction:       {summary['consistency_pass_percent']:.3f} %")
print(f"estimated vortex-centre error:   {summary['center_error_px']:.6f} px")
print(f"recovered core-radius proxy:     {recovered_core_radius_px:.6f} px")
print()
print("STAGE COMPARISON")
print(f"raw Farneback median EPE:        {summary['raw_median_epe_px']:.6f} px/frame")
print(f"processed median EPE:            {summary['processed_median_epe_px']:.6f} px/frame")
print("============================================================")

SUMMARY_PATH = RESULT_DIR / "rankine_validation_summary.txt"
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("2WCF synthetic Rankine-vortex validation\n")
    f.write(f"center_x = {center_x:.8f}\n")
    f.write(f"center_y = {center_y:.8f}\n")
    f.write(f"core_radius_px = {core_radius_px:.8f}\n")
    f.write(f"boundary_tangential_speed_px = {boundary_tangential_speed_px:.8f}\n")
    f.write(f"omega_core_rad_per_frame = {omega_core_rad_per_frame:.8f}\n")
    f.write(f"frame_pairs = {len(records)}\n")
    f.write(f"interior_border = {INTERIOR_BORDER_PX}\n")
    f.write(f"recovered_core_radius_proxy_px = {recovered_core_radius_px}\n\n")
    for key, value in summary.items():
        f.write(f"{key} = {value}\n")


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================
frame_pairs = np.array([r["frame_pair"] for r in records])
median_epe = np.array([r["final_median_epe_px"] for r in records])
p95_epe = np.array([r["final_p95_epe_px"] for r in records])
retention = np.array([r["interior_retention_percent"] for r in records])
radial_bias = np.array([r["final_median_radial_bias_px"] for r in records])
tangential_bias = np.array([r["final_median_tangential_bias_px"] for r in records])
center_error = np.array([r["center_error_px"] for r in records])

plt.figure(figsize=(8, 5))
plt.plot(frame_pairs, median_epe, label="Median EPE")
plt.plot(frame_pairs, p95_epe, label="95th percentile EPE")
plt.xlabel("Frame pair")
plt.ylabel("Endpoint error [px/frame]")
plt.title("Rankine-vortex optical-flow error")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "epe_vs_frame.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(frame_pairs, retention)
plt.xlabel("Frame pair")
plt.ylabel("Interior retained vectors [%]")
plt.title("2WCF vector retention")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "retention_vs_frame.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(frame_pairs, radial_bias, label="Median radial bias")
plt.plot(frame_pairs, tangential_bias, label="Median tangential bias")
plt.axhline(0.0, linewidth=1)
plt.xlabel("Frame pair")
plt.ylabel("Component bias [px/frame]")
plt.title("Polar-component bias")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "polar_component_bias_vs_frame.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(frame_pairs, center_error)
plt.xlabel("Frame pair")
plt.ylabel("Centre error [px]")
plt.title("Recovered vortex-centre error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "center_error_vs_frame.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(radius_centers, retention_vs_radius)
plt.axvline(core_radius_px, linewidth=1)
plt.xlabel("Radius [px]")
plt.ylabel("Retained vectors [%]")
plt.title("Retention versus radius")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "retention_vs_radius.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(radius_centers, median_epe_vs_radius)
plt.axvline(core_radius_px, linewidth=1)
plt.xlabel("Radius [px]")
plt.ylabel("Median EPE [px/frame]")
plt.title("Median endpoint error versus radius")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "epe_vs_radius.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(radius_centers, median_true_t, label="True tangential displacement")
plt.plot(radius_centers, median_est_t, label="Estimated tangential displacement")
plt.axvline(core_radius_px, linewidth=1)
plt.xlabel("Radius [px]")
plt.ylabel("Tangential displacement [px/frame]")
plt.title("Tangential displacement profile")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "tangential_profile_vs_radius.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(radius_centers, median_radial_bias_vs_radius, label="Median radial bias")
plt.plot(radius_centers, median_tangential_bias_vs_radius, label="Median tangential bias")
plt.axhline(0.0, linewidth=1)
plt.axvline(core_radius_px, linewidth=1)
plt.xlabel("Radius [px]")
plt.ylabel("Bias [px/frame]")
plt.title("Bias versus radius")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "bias_vs_radius.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(radius_centers, median_abs_t_error_vs_radius)
plt.axvline(core_radius_px, linewidth=1)
plt.xlabel("Radius [px]")
plt.ylabel("Median |tangential error| [px/frame]")
plt.title("Tangential error versus radius")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "tangential_error_vs_radius.png", dpi=200)
plt.close()

# Spatial median EPE over time, on the sampled grid.
spatial_epe_stack = np.stack(spatial_epe_stack, axis=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    spatial_median_epe = np.nanmedian(spatial_epe_stack, axis=0)

plt.figure(figsize=(7, 6))
image = plt.imshow(
    spatial_median_epe,
    origin="upper",
    extent=[reference_x.min(), reference_x.max(), reference_y.max(), reference_y.min()],
    aspect="equal",
)
plt.colorbar(image, label="Median EPE [px/frame]")
plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.title("Spatial median endpoint error")
plt.tight_layout()
plt.savefig(RESULT_DIR / "spatial_median_epe.png", dpi=200)
plt.close()

print(f"\nSaved per-frame CSV: {CSV_PATH}")
print(f"Saved radial profile: {RADIAL_CSV_PATH}")
print(f"Saved summary:        {SUMMARY_PATH}")
print(f"Saved plots to:       {RESULT_DIR}")
