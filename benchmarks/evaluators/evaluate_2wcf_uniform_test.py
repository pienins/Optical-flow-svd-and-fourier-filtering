from pathlib import Path
import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# USER SETTINGS
# =============================================================================
# Folder created by generate_uniform_flow_test.py
SYNTHETIC_TEST_DIR = Path(r"E:/Ansis Zivers/Optical Flow/Image_sequences/SYNTHETIC/Uniform_flow/synthetic_uniform_u1_v0")

# Set this to the output folder created by 2WCF-SVD_v4_5_fixed.py.
# It must contain a subfolder called "validation".
PIPELINE_OUTPUT_DIR = Path(r"E:/Ansis Zivers/Optical Flow/Output_2WCF_SVD/SYNTHETIC/frames_run_1")

# Report both the complete sampled field and an interior region. The interior
# metric prevents border handling from hiding the underlying OF accuracy.
INTERIOR_BORDER_PX = 24
# =============================================================================


GT_PATH = SYNTHETIC_TEST_DIR / "true_vector_values.npz"
VALIDATION_DIR = PIPELINE_OUTPUT_DIR / "validation"
RESULT_DIR = PIPELINE_OUTPUT_DIR / "synthetic_validation_results"
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
    true_u = float(gt["true_u"])
    true_v = float(gt["true_v"])
    true_speed = float(gt["true_speed"])

HEIGHT, WIDTH = u_gt_full.shape


def endpoint_error(u, v, u_truth, v_truth):
    return np.sqrt((u - u_truth) ** 2 + (v - v_truth) ** 2)


def angular_error_deg(u, v, u_truth, v_truth):
    est_mag = np.sqrt(u**2 + v**2)
    gt_mag = np.sqrt(u_truth**2 + v_truth**2)
    denom = est_mag * gt_mag

    out = np.full_like(est_mag, np.nan, dtype=np.float64)
    usable = denom > np.finfo(np.float64).eps

    cosang = np.empty_like(est_mag, dtype=np.float64)
    cosang[usable] = (u[usable] * u_truth[usable] + v[usable] * v_truth[usable]) / denom[usable]
    cosang[usable] = np.clip(cosang[usable], -1.0, 1.0)
    out[usable] = np.degrees(np.arccos(cosang[usable]))
    return out


def safe_median(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def safe_mean(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def safe_percentile(x, q):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


records = []

# For an across-time spatial error map.
spatial_epe_stack = []
spatial_valid_stack = []
reference_x = None
reference_y = None

for pair_index, path in enumerate(validation_files):
    with np.load(path) as data:
        x = data["x"].astype(np.int64)
        y = data["y"].astype(np.int64)

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

    if reference_x is None:
        reference_x = x.copy()
        reference_y = y.copy()
    elif not (np.array_equal(x, reference_x) and np.array_equal(y, reference_y)):
        raise ValueError("Sample grid changed between frame pairs; cannot aggregate spatial maps safely.")

    u_truth = u_gt_full[y, x]
    v_truth = v_gt_full[y, x]

    interior = (
        (x >= INTERIOR_BORDER_PX) &
        (x < WIDTH - INTERIOR_BORDER_PX) &
        (y >= INTERIOR_BORDER_PX) &
        (y < HEIGHT - INTERIOR_BORDER_PX)
    )

    # Raw Farneback accuracy before rejection/smoothing.
    epe_raw = endpoint_error(u_raw, v_raw, u_truth, v_truth)

    # Processed field before final sign inversion/rejection.
    epe_processed = endpoint_error(u_processed, v_processed, u_truth, v_truth)

    # Final output field from the pipeline. Accuracy is meaningful only where retained.
    epe_output = endpoint_error(u_output, v_output, u_truth, v_truth)
    angle_output = angular_error_deg(u_output, v_output, u_truth, v_truth)

    final = valid_mask
    final_interior = valid_mask & interior

    total = int(valid_mask.size)
    retained = int(np.count_nonzero(final))
    interior_total = int(np.count_nonzero(interior))
    interior_retained = int(np.count_nonzero(final_interior))

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

        "final_median_u_px": safe_median(u_output[final_interior]),
        "final_median_v_px": safe_median(v_output[final_interior]),
        "final_mean_u_px": safe_mean(u_output[final_interior]),
        "final_mean_v_px": safe_mean(v_output[final_interior]),
        "final_u_bias_median_px": safe_median(u_output[final_interior] - u_truth[final_interior]),
        "final_v_bias_median_px": safe_median(v_output[final_interior] - v_truth[final_interior]),
        "final_median_epe_px": safe_median(epe_output[final_interior]),
        "final_mean_epe_px": safe_mean(epe_output[final_interior]),
        "final_p95_epe_px": safe_percentile(epe_output[final_interior], 95),
        "final_median_relative_epe_percent": (
            100.0 * safe_median(epe_output[final_interior]) / true_speed
            if true_speed > 0 else float("nan")
        ),
        "final_median_angle_error_deg": safe_median(angle_output[final_interior]),
        "median_fb_consistency_error_px": safe_median(consistency_error[interior]),
        "median_confidence": safe_median(confidence[interior]),
        "consistency_pass_percent": 100.0 * np.count_nonzero(consistency_mask & interior) / max(interior_total, 1),
    }
    records.append(record)

    epe_for_map = np.where(valid_mask, epe_output, np.nan)
    spatial_epe_stack.append(epe_for_map)
    spatial_valid_stack.append(valid_mask)


# =============================================================================
# SAVE PER-FRAME CSV
# =============================================================================
CSV_PATH = RESULT_DIR / "uniform_validation_per_frame.csv"
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

print("\n============================================================")
print("2WCF SYNTHETIC UNIFORM-FLOW VALIDATION")
print("============================================================")
print(f"Ground truth u:       {true_u:.6f} px/frame")
print(f"Ground truth v:       {true_v:.6f} px/frame")
print(f"Ground truth speed:   {true_speed:.6f} px/frame")
print(f"Frame pairs analysed: {len(records)}")
print(f"Interior border:      {INTERIOR_BORDER_PX} px")
print()
print("FINAL PIPELINE OUTPUT — median across frame pairs")
print(f"u measured:                 {summary['final_median_u_px']:.6f} px/frame")
print(f"v measured:                 {summary['final_median_v_px']:.6f} px/frame")
print(f"median u bias:              {summary['final_u_bias_median_px']:.6f} px/frame")
print(f"median v bias:              {summary['final_v_bias_median_px']:.6f} px/frame")
print(f"median endpoint error:      {summary['final_median_epe_px']:.6f} px/frame")
print(f"mean endpoint error:        {summary['final_mean_epe_px']:.6f} px/frame")
print(f"95th-percentile EPE:        {summary['final_p95_epe_px']:.6f} px/frame")
print(f"median relative EPE:        {summary['final_median_relative_epe_percent']:.3f} %")
print(f"median angular error:       {summary['final_median_angle_error_deg']:.4f} deg")
print(f"interior retention:         {summary['interior_retention_percent']:.3f} %")
print(f"consistency pass fraction:  {summary['consistency_pass_percent']:.3f} %")
print()
print("STAGE COMPARISON")
print(f"raw Farneback median EPE:   {summary['raw_median_epe_px']:.6f} px/frame")
print(f"processed median EPE:       {summary['processed_median_epe_px']:.6f} px/frame")
print("============================================================")

SUMMARY_PATH = RESULT_DIR / "uniform_validation_summary.txt"
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("2WCF synthetic uniform-flow validation\n")
    f.write(f"True u = {true_u:.8f} px/frame\n")
    f.write(f"True v = {true_v:.8f} px/frame\n")
    f.write(f"True speed = {true_speed:.8f} px/frame\n")
    f.write(f"Frame pairs = {len(records)}\n")
    f.write(f"Interior border = {INTERIOR_BORDER_PX} px\n\n")
    for key, value in summary.items():
        f.write(f"{key} = {value}\n")


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================
frame_pairs = np.array([r["frame_pair"] for r in records])
median_epe = np.array([r["final_median_epe_px"] for r in records])
p95_epe = np.array([r["final_p95_epe_px"] for r in records])
retention = np.array([r["interior_retention_percent"] for r in records])
u_bias = np.array([r["final_u_bias_median_px"] for r in records])
v_bias = np.array([r["final_v_bias_median_px"] for r in records])

plt.figure(figsize=(8, 5))
plt.plot(frame_pairs, median_epe, label="Median EPE")
plt.plot(frame_pairs, p95_epe, label="95th percentile EPE")
plt.xlabel("Frame pair")
plt.ylabel("Endpoint error [px/frame]")
plt.title("Uniform-flow optical-flow error")
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
plt.plot(frame_pairs, u_bias, label="u bias")
plt.plot(frame_pairs, v_bias, label="v bias")
plt.axhline(0.0, linewidth=1)
plt.xlabel("Frame pair")
plt.ylabel("Median component bias [px/frame]")
plt.title("Velocity-component bias")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "component_bias_vs_frame.png", dpi=200)
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
print(f"Saved summary:       {SUMMARY_PATH}")
print(f"Saved plots to:      {RESULT_DIR}")
