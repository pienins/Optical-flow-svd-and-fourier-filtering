"""Evaluate a 2WCF run against the homogeneous (uniform) translation ground truth.

Reads the ground truth written by generate.py and the validation_*.npz files
of one pipeline run (diagnostics.SaveValidationData = true), matches vectors at
their sampled pixel coordinates, and writes per-pair metrics, a summary and
plots to <pipeline_output_dir>/synthetic_validation_results/.

Usage (from any directory):
    python benchmarks/homogeneous_constant_flow/evaluate.py [--config PATH]

The same config.json as generate.py is used, so the ground-truth folder is
derived from the flow parameters; only paths.pipeline_output_dir must point
to the run directory produced by run.py.
"""
import argparse
import csv
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # benchmarks/, for common.metrics

from common.metrics import (  # noqa: E402
    angular_error_deg, endpoint_error, safe_mean, safe_median, safe_percentile,
)

# Load this scenario's generate.py under a unique module name. Every scenario
# has a generate.py, so a plain "import generate" could pick up another one.
_spec = importlib.util.spec_from_file_location(
    "homogeneous_constant_flow_generate", HERE / "generate.py")
_generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generate)
DEFAULT_CONFIG = _generate.DEFAULT_CONFIG
load_config = _generate.load_config
sequence_dir = _generate.sequence_dir


def recorded_velocity_inversion(run_dir):
    """Return VelocityInversion recorded in the run's parameters.json, or None.

    Current runs store grouped parameters; older runs stored a flat dict.
    """
    path = run_dir / "parameters.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        params = json.load(f)
    if "VelocityInversion" in params:
        return params["VelocityInversion"]
    for group in params.values():
        if isinstance(group, dict) and "VelocityInversion" in group:
            return group["VelocityInversion"]
    return None


def evaluate(cfg):
    gt_path = sequence_dir(cfg) / "true_vector_values.npz"
    pipeline_output_dir = cfg["paths"]["pipeline_output_dir"]
    interior_border_px = cfg["evaluate"]["interior_border_px"]
    validation_dir = pipeline_output_dir / "validation"
    result_dir = pipeline_output_dir / "synthetic_validation_results"

    if not gt_path.exists():
        raise FileNotFoundError(f"True vector file not found: {gt_path}")

    if not validation_dir.exists():
        raise FileNotFoundError(
            f"Validation folder not found: {validation_dir}\n"
            "Make sure diagnostics.SaveValidationData = true in the 2WCF config."
        )

    validation_files = sorted(validation_dir.glob("validation_*.npz"))
    if not validation_files:
        raise FileNotFoundError(f"No validation_*.npz files found in {validation_dir}")

    # u_output already has VelocityInversion applied. Comparing it with truth in
    # image coordinates is only meaningful when the sign is unchanged.
    inversion = recorded_velocity_inversion(pipeline_output_dir)
    if inversion is None:
        warnings.warn("Run parameters.json not found or lacks VelocityInversion; "
                      "assuming VelocityInversion = 1.")
    elif inversion != 1:
        raise ValueError(
            f"Run used VelocityInversion = {inversion}. u_output/v_output are then "
            "not in image coordinates (+u right, +v down) and cannot be compared "
            "with the ground truth. Re-run with VelocityInversion = 1."
        )

    result_dir.mkdir(parents=True, exist_ok=True)

    with np.load(gt_path) as gt:
        u_gt_full = gt["u_gt"].astype(np.float64)
        v_gt_full = gt["v_gt"].astype(np.float64)
        true_u = float(gt["true_u"])
        true_v = float(gt["true_v"])
        true_speed = float(gt["true_speed"])

    height, width = u_gt_full.shape

    records = []

    # For an across-time spatial error map.
    spatial_epe_stack = []
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
            if x.max() >= width or y.max() >= height:
                raise ValueError(
                    f"Sample coordinates exceed the {width}x{height} ground truth; "
                    "the run does not belong to this generated sequence."
                )
        elif not (np.array_equal(x, reference_x) and np.array_equal(y, reference_y)):
            raise ValueError("Sample grid changed between frame pairs; cannot aggregate spatial maps safely.")

        u_truth = u_gt_full[y, x]
        v_truth = v_gt_full[y, x]

        interior = (
            (x >= interior_border_px) &
            (x < width - interior_border_px) &
            (y >= interior_border_px) &
            (y < height - interior_border_px)
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

        spatial_epe_stack.append(np.where(valid_mask, epe_output, np.nan))

    # =========================================================================
    # PER-FRAME CSV
    # =========================================================================
    csv_path = result_dir / "uniform_validation_per_frame.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # =========================================================================
    # AGGREGATED SUMMARY (median across frame pairs)
    # =========================================================================
    summary = {}
    for key in records[0].keys():
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
    print(f"Interior border:      {interior_border_px} px")
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

    summary_path = result_dir / "uniform_validation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("2WCF synthetic uniform-flow validation\n")
        f.write(f"True u = {true_u:.8f} px/frame\n")
        f.write(f"True v = {true_v:.8f} px/frame\n")
        f.write(f"True speed = {true_speed:.8f} px/frame\n")
        f.write(f"Frame pairs = {len(records)}\n")
        f.write(f"Interior border = {interior_border_px} px\n\n")
        for key, value in summary.items():
            f.write(f"{key} = {value}\n")

    # =========================================================================
    # DIAGNOSTIC PLOTS
    # =========================================================================
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
    plt.savefig(result_dir / "epe_vs_frame.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(frame_pairs, retention)
    plt.xlabel("Frame pair")
    plt.ylabel("Interior retained vectors [%]")
    plt.title("2WCF vector retention")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(result_dir / "retention_vs_frame.png", dpi=200)
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
    plt.savefig(result_dir / "component_bias_vs_frame.png", dpi=200)
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
    plt.savefig(result_dir / "spatial_median_epe.png", dpi=200)
    plt.close()

    print(f"\nSaved per-frame CSV: {csv_path}")
    print(f"Saved summary:       {summary_path}")
    print(f"Saved plots to:      {result_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="benchmark config.json")
    args = parser.parse_args(argv)
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
