# -*- coding: utf-8 -*-
"""
2WCF-SVD_v4_4.py

Two-way consistent Farnebäck optical flow + optional temporal SVD filtering,
with timing logs, optional parallelization, and confidence-based flow smoothing
from forward/backward consistency.

Modes:
- If UseConfidenceSmoothing = True:
    * use soft confidence c = exp(-(err/σ_frame)^2) with σ_frame = median_err + ConfidenceSigma * MAD,
      confidence-weighted local smoothing, and MinConfidence cutoff.
- If UseConfidenceSmoothing = False:
    * use adaptive hard thresholding based on robust statistics:
      thr_frame = median_err + HardThresholdK * MAD, with ConsistencyErrorThreshold as a fallback.

Refactored from Two_way_consistent_farneback_w_fourier_svd_v2.py
@main_dev: Ansis_Z
@co_dev:   Mihails_B
"""

import os
import re
import time
import cv2
import json
import pickle
import imageio.v2 as imageio
import numpy as np
import pandas
import matplotlib.pyplot as plt

from io import BytesIO
from tifffile import imread
from scipy.fft import fft, fftfreq
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import wolframclient.serializers as wxf

#%%

# =============================================================================
# PARAMETERS (updated from parameters.json and extended)
# =============================================================================

# --- High-level switches ---
USE_SVD = False              # If False: skip SVD and use raw TIFFs
UseParallelFlow = True      # Parallelise Farnebäck across frame pairs
MaxFlowWorkers = None       # None -> use (cpu_count - 1) if available

# --- Confidence-based smoothing parameters ---
UseConfidenceSmoothing = True  # If False: use adaptive hard thresholding instead of confidence smoothing
ConfidenceSigma = 5           # k_sigma: σ_frame = median_err + k_sigma * MAD (dimensionless factor)
ConfidenceKernelSize = 5        # Neighborhood size for confidence-weighted smoothing (odd)
MinConfidence = 0.01           # Drop vectors below this confidence when sampling (soft mode)

# --- Visualization parameters ---
ArrowDensityStep = 7            # from parameters.json
ArrowScale = 50               # controls arrow size inversely in quiver
VideoFPS = 5                    # from parameters.json
VideoDPI = 200                
VelocityInversion = -1          # Flip direction if vectors appear inverted

# --- Optical flow parameters (from parameters.json) ---
# These map directly to cv2.calcOpticalFlowFarneback(...):
#   PyramidDownsampleFactor -> pyr_scale  : image scale (<1) between pyramid levels
#   PyramidLevels           -> levels     : number of pyramid layers (including original)
#   WindowSize              -> winsize    : averaging window size; larger = smoother, more robust, but blurrier flow
#   IterationCount          -> iterations : number of iterations at each pyramid level
#   PolyFitKernelSize       -> poly_n     : size of the pixel neighborhood used for polynomial expansion
#   GaussKernelStdev        -> poly_sigma : std of Gaussian used for polynomial smoothing; larger = smoother, less detail
#   FilterFlags             -> flags      : algorithm flags (cv2.OPTFLOW_FARNEBACK_* constants)
PyramidDownsampleFactor = 0.75
PyramidLevels = 8
WindowSize = 25
IterationCount = 10
PolyFitKernelSize = 7
GaussKernelStdev = 2
FilterFlags = 0

# Fallback hard threshold for consistency; used if robust stats fail (NaNs)
ConsistencyErrorThreshold = 1.0

# Frame-adaptive hard threshold factor (thr = med + HardThresholdK * MAD)
HardThresholdK = 3.0

# Group size for median-of-means robust statistic
MedianOfMeansGroupSize = 10000


# --- Temporal SVD modes to remove (from parameters.json) ---
KillIDs = [
    0
]

# --- Run ID (careful with output overwrites) ---
runID = 2

#%%

# =============================================================================
# UTILS
# =============================================================================

def numeric_sort_key(filename: str) -> int:
    """Extract first integer from filename for natural sorting."""
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else -1


# =============================================================================
# SVD FILTERING
# =============================================================================

def svd_filter_images(image_folder: str, remove_modes, do_plots: bool = True):
    """
    Load a stack of TIFF images, perform temporal SVD across frames,
    remove selected temporal modes and return the filtered volume.
    """
    print("\n[ SVD ] Loading TIFF images...")
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(".tiff")],
        key=numeric_sort_key,
    )

    if len(image_files) == 0:
        raise FileNotFoundError(f"No .tiff images found in folder: {image_folder}")

    images = [
        imread(os.path.join(image_folder, f))
        for f in tqdm(image_files, desc="Reading TIFFs")
    ]
    volume = np.stack(images, axis=0).astype(np.float32)  # shape: (T, H, W)
    T, H, W = volume.shape
    print(f"[ SVD ] Loaded image stack shape: (T={T}, H={H}, W={W})")

    # Flatten (T, H*W) for SVD
    print("\n[ SVD ] Performing SVD...")
    X = volume.reshape(T, -1)  # shape: (T, Npix)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    if do_plots:
        # Temporal modes (first few)
        plt.figure(figsize=(8, 3))
        for i in range(min(3, U.shape[1])):
            plt.plot(U[:, i], label=f"Mode {i}")
        plt.title("Temporal SVD Modes")
        plt.xlabel("Frame")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Frequency analysis for the first few modes
        num_modes_to_plot = min(10, U.shape[1])
        fig, axs = plt.subplots(
            num_modes_to_plot, 1,
            figsize=(10, 2 * num_modes_to_plot),
            sharex=True
        )
        fig.suptitle("FFT of Temporal SVD Modes")

        for i in range(num_modes_to_plot):
            signal = U[:, i]
            N = len(signal)
            yf = np.abs(fft(signal - np.mean(signal)))
            xf = fftfreq(N, d=1)[:N // 2]
            axs[i].plot(xf, yf[:N // 2])
            axs[i].set_ylabel(f"Mode {i}")
            axs[i].grid(True)

        axs[-1].set_xlabel("Frequency (1/frame)")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    # Remove selected modes
    print("\n[ SVD ] Removing selected modes:", remove_modes)
    recon = np.zeros_like(X)

    # We reconstruct only the unwanted contribution, then subtract it.
    for i in tqdm(remove_modes, desc="Removing modes"):
        if i < 0 or i >= U.shape[1]:
            print(f"  Warning: mode index {i} is out of range, skipping.")
            continue
        recon += np.outer(U[:, i], S[i] * Vt[i, :])

    X_filtered = X - recon
    filtered_volume = X_filtered.reshape(T, H, W)

    print(f"[ SVD ] Filtered volume stats: min={X_filtered.min():.3f}, "
          f"max={X_filtered.max():.3f}")

    return filtered_volume, image_files


def load_images_without_svd(image_folder: str):
    """
    Load and return raw TIFF images as a list of 2D arrays (float32),
    along with sorted file names.
    """
    print("\n[ LOAD ] Loading TIFF images without SVD...")
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(".tiff")],
        key=numeric_sort_key,
    )
    if len(image_files) == 0:
        raise FileNotFoundError(f"No .tiff images found in folder: {image_folder}")

    images = [
        imread(os.path.join(image_folder, f)).astype(np.float32)
        for f in tqdm(image_files, desc="Reading TIFFs")
    ]
    T = len(images)
    H, W = images[0].shape
    print(f"[ LOAD ] Loaded image stack shape: (T={T}, H={H}, W={W})")
    return images, image_files


# =============================================================================
# OPTICAL FLOW
# =============================================================================

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize frame to uint8 [0, 255] for optical flow input.
    """
    frame = frame.astype(np.float32)
    frame = frame - np.min(frame)  # shift min to 0
    max_val = np.max(frame)
    if max_val > 0:
        frame = frame / max_val  # scale to [0,1]
    frame_uint8 = (frame * 255).astype(np.uint8)
    return frame_uint8


def median_of_means(x: np.ndarray, group_size: int = MedianOfMeansGroupSize) -> float:
    """
    Robust statistic: split x into groups of size `group_size`,
    compute mean in each group, then return the median of those means.
    Very robust to outliers for large arrays.
    """
    x = np.asarray(x)
    # Keep only finite values
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    g = max(1, n // group_size)
    trimmed = x[:g * group_size]
    if trimmed.size == 0:
        return float("nan")
    groups = trimmed.reshape(g, group_size)
    group_means = groups.mean(axis=1)
    return float(np.median(group_means))


def confidence_weighted_smoothing(u: np.ndarray,
                                  v: np.ndarray,
                                  consistency_error: np.ndarray,
                                  sigma: float) -> tuple:
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
    # Ensure strictly positive sigma
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1e-6

    # Confidence in [0, 1]
    conf = np.exp(- (consistency_error / sigma) ** 2).astype(np.float32)

    k = ConfidenceKernelSize
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1  # enforce odd kernel size

    # Weighted averages using box filter (fast convolution)
    weight_sum = cv2.boxFilter(conf, ddepth=-1, ksize=(k, k), normalize=False)

    u_weighted = cv2.boxFilter(u * conf, ddepth=-1, ksize=(k, k), normalize=False)
    v_weighted = cv2.boxFilter(v * conf, ddepth=-1, ksize=(k, k), normalize=False)

    eps = 1e-6
    u_s = u_weighted / (weight_sum + eps)
    v_s = v_weighted / (weight_sum + eps)

    return u_s.astype(np.float32), v_s.astype(np.float32), conf


def compute_flow_pair(index: int, images, h: int, w: int):
    """
    Compute two-way-consistent Farneback flow for frame pair (index, index+1)
    and return sampled velocity vectors plus robust consistency diagnostics.

    Modes:
    - UseConfidenceSmoothing = True:
        * forward/backward Farneback
        * consistency error -> confidence map -> smoothed flow
        * soft MinConfidence threshold
    - UseConfidenceSmoothing = False:
        * forward/backward Farneback
        * legacy hard mask, but with adaptive per-frame threshold:
          thr = median(consistency_error) + HardThresholdK * MAD
    """
    prev_img = normalize_frame(images[index])
    next_img = normalize_frame(images[index + 1])

    # Forward flow
    flow_fwd = cv2.calcOpticalFlowFarneback(
        prev_img, next_img, None,
        PyramidDownsampleFactor,
        PyramidLevels,
        WindowSize,
        IterationCount,
        PolyFitKernelSize,
        GaussKernelStdev,
        FilterFlags,
    )
    u = flow_fwd[..., 0]
    v = flow_fwd[..., 1]

    # Backward flow for consistency
    flow_bwd = cv2.calcOpticalFlowFarneback(
        next_img, prev_img, None,
        PyramidDownsampleFactor,
        PyramidLevels,
        WindowSize,
        IterationCount,
        PolyFitKernelSize,
        GaussKernelStdev,
        FilterFlags,
    )

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    coords_x = (grid_x + u).astype(np.float32)
    coords_y = (grid_y + v).astype(np.float32)

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
    consistency_error = np.sqrt((u + bwd_u_warped) ** 2 +
                                (v + bwd_v_warped) ** 2)

    # --- robust error stats (per frame pair) ---
    err_flat = consistency_error.ravel()
    err_flat = err_flat[np.isfinite(err_flat)]

    if err_flat.size > 0:
        med = float(np.median(err_flat))
        mad_raw = float(np.median(np.abs(err_flat - med)))
        mad = 1.4826 * mad_raw  # scaled MAD (≈ std for Gaussian)
        mom = median_of_means(err_flat, group_size=MedianOfMeansGroupSize)
    else:
        med = float("nan")
        mad = float("nan")
        mom = float("nan")

    # Frame-adaptive threshold and sigma (even if we don't always use both)
    if np.isfinite(med) and np.isfinite(mad):
        thr_frame = med + HardThresholdK * mad
        sigma_frame = med + ConfidenceSigma * mad
    else:
        thr_frame = ConsistencyErrorThreshold
        sigma_frame = 1.0

    # Enforce positivity and numerical sanity
    if not np.isfinite(thr_frame):
        thr_frame = ConsistencyErrorThreshold
    if not np.isfinite(sigma_frame) or sigma_frame <= 0:
        sigma_frame = 1e-6

    # Sample grid (same in both modes)
    y_grid, x_grid = np.mgrid[
        ArrowDensityStep // 2:h:ArrowDensityStep,
        ArrowDensityStep // 2:w:ArrowDensityStep,
    ]

    if UseConfidenceSmoothing:
        # ---- Soft mode: confidence + smoothing + MinConfidence ----
        u_s, v_s, conf_map = confidence_weighted_smoothing(u, v, consistency_error, sigma_frame)

        u_sampled = u_s[
            ArrowDensityStep // 2:h:ArrowDensityStep,
            ArrowDensityStep // 2:w:ArrowDensityStep,
        ]
        v_sampled = v_s[
            ArrowDensityStep // 2:h:ArrowDensityStep,
            ArrowDensityStep // 2:w:ArrowDensityStep,
        ]
        conf_sampled = conf_map[
            ArrowDensityStep // 2:h:ArrowDensityStep,
            ArrowDensityStep // 2:w:ArrowDensityStep,
        ]

        valid_indices = conf_sampled >= MinConfidence

    else:
        # ---- Adaptive hard thresholding mode ----
        reliability_mask = consistency_error < thr_frame

        u_sampled = u[
            ArrowDensityStep // 2:h:ArrowDensityStep,
            ArrowDensityStep // 2:w:ArrowDensityStep,
        ]
        v_sampled = v[
            ArrowDensityStep // 2:h:ArrowDensityStep,
            ArrowDensityStep // 2:w:ArrowDensityStep,
        ]
        mask_sampled = reliability_mask[
            ArrowDensityStep // 2:h:ArrowDensityStep,
            ArrowDensityStep // 2:w:ArrowDensityStep,
        ]

        valid_indices = mask_sampled

    x_valid = x_grid[valid_indices]
    y_valid = y_grid[valid_indices]

    u_valid = VelocityInversion * u_sampled[valid_indices]
    v_valid = VelocityInversion * v_sampled[valid_indices]

    return {
        "index": index,
        "x": x_valid.astype(float),
        "y": y_valid.astype(float),
        "u": u_valid.astype(float),
        "v": v_valid.astype(float),
        # diagnostics
        "median_err": med,
        "mad_err": mad,
        "mom_err": mom,
        "thr_frame": float(thr_frame),
        "sigma_frame": float(sigma_frame),
    }


def run_optical_flow(images,
                     output_folder: str,
                     output_pkl: str,
                     output_wxf: str):
    """
    Run two-way-consistent Farneback optical flow on a list of images.

    It:
    - computes two-way-consistent Farnebäck flow for each frame pair,
    - applies either confidence-based smoothing (soft mode) or adaptive hard thresholding,
    - saves velocity fields (PKL + WXF) and an MP4 visualization,
    - collects robust consistency diagnostics per frame pair (median_err, mad_err, etc.),
    - writes those diagnostics to flow_qc_metrics.csv and a PNG plot in `output_folder`.
    """
    print("\n[ FLOW ] Starting optical flow analysis...")

    if len(images) < 2:
        print("[ FLOW ] Not enough frames for optical flow.")
        return

    h, w = images[0].shape
    indices = list(range(len(images) - 1))

    # -------------------------------------------------------------------------
    # Stage 1: compute flows (parallelizable)
    # -------------------------------------------------------------------------
    start_flow = time.perf_counter()
    results = {}

    if UseParallelFlow and len(indices) > 1:
        cpu_count = os.cpu_count() or 1
        workers = MaxFlowWorkers or max(1, cpu_count - 1)
        print(f"[ FLOW ] Running Farneback in parallel with {workers} workers...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(compute_flow_pair, i, images, h, w): i
                for i in indices
            }

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(indices),
                desc="Farneback + consistency + smoothing/threshold",
            ):
                res = future.result()
                results[res["index"]] = res
    else:
        print("[ FLOW ] Running Farneback sequentially...")
        for i in tqdm(indices, desc="Farneback + consistency + smoothing/threshold"):
            res = compute_flow_pair(i, images, h, w)
            results[res["index"]] = res

    flow_elapsed = time.perf_counter() - start_flow
    avg_per_pair = flow_elapsed / max(1, len(indices))
    print(f"[ FLOW ] Flow computation: {flow_elapsed:.2f} s total "
          f"({avg_per_pair:.2f} s per frame pair).")

    # -------------------------------------------------------------------------
    # Stage 2: save fields, generate quiver frames, build MP4
    #         and collect QC diagnostics
    # -------------------------------------------------------------------------
    #video_path = os.path.join(output_folder, "velocity_field.mp4")
    video_path = os.path.join(output_folder, "velocity_field.avi")
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = None
    frame_size = None

    qc_records = []

    print("[ FLOW ] Saving velocity fields, composing MP4, and collecting QC metrics...")

    for i in tqdm(indices, desc="Saving fields + video"):
        res = results[i]
        x = res["x"]
        y = res["y"]
        u_vec = res["u"]
        v_vec = res["v"]

        # Save velocity field as PKL + WXF
        flat_data = {
            "x": x.ravel(),
            "y": y.ravel(),
            "u": u_vec.ravel(),
            "v": v_vec.ravel(),
        }

        pkl_path = os.path.join(output_pkl, f"velocity_field_{i:05d}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(flat_data, f, pickle.HIGHEST_PROTOCOL)

        wxf_path = os.path.join(output_wxf, f"velocity_field_{i:05d}.wxf")
        wxf.export(flat_data, wxf_path, target_format="wxf")

        # Collect QC metrics for this frame pair
        qc_records.append({
            "frame_index": i,
            "median_err": res.get("median_err", float("nan")),
            "mad_err": res.get("mad_err", float("nan")),
            "mom_err": res.get("mom_err", float("nan")),
            "thr_frame": res.get("thr_frame", float("nan")),
            "sigma_frame": res.get("sigma_frame", float("nan")),
            "use_confidence_smoothing": bool(UseConfidenceSmoothing),
        })

        # Plot and save frame (sequential; Matplotlib is not thread-safe)
        prev_img = normalize_frame(images[i])

        fig, ax = plt.subplots(figsize=(6, 6), dpi=VideoDPI)
        ax.imshow(prev_img, cmap="gray")
        ax.quiver(x, y, u_vec, v_vec, color="lime", scale=ArrowScale)
        ax.set_title(f"Frame {i} → {i + 1}")
        ax.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        frame = imageio.imread(buf)
        buf.close()

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        if video_writer is None:
            frame_size = (frame.shape[1], frame.shape[0])
            video_writer = cv2.VideoWriter(video_path, fourcc, VideoFPS, frame_size)
            if not video_writer.isOpened():
                raise IOError(f"Failed to open video writer at: {video_path}")

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(frame_bgr, frame_size)
        video_writer.write(resized)

    if video_writer is not None:
        video_writer.release()

    print(f"[ FLOW ] Optical flow MP4 saved to: {video_path}")

    # -------------------------------------------------------------------------
    # Stage 3: write QC CSV
    # -------------------------------------------------------------------------
    if qc_records:
        qc_df = pandas.DataFrame(qc_records)
        qc_csv_path = os.path.join(output_folder, "flow_qc_metrics.csv")
        qc_df.to_csv(qc_csv_path, index=False)
        print(f"[ FLOW ] QC metrics saved to: {qc_csv_path}")

    # -------------------------------------------------------------------------
    # Stage 4: plot median_err and mad_err vs frame index
    # -------------------------------------------------------------------------
    if qc_records:
        try:
            frames = [rec["frame_index"] for rec in qc_records]
            medians = [rec["median_err"] for rec in qc_records]
            mads = [rec["mad_err"] for rec in qc_records]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            fig.suptitle("Forward–Backward Consistency Diagnostics")

            ax1.plot(frames, medians, marker="o")
            ax1.set_ylabel("median_err")
            ax1.grid(True, alpha=0.3)

            ax2.plot(frames, mads, marker="o")
            ax2.set_xlabel("frame index")
            ax2.set_ylabel("mad_err")
            ax2.grid(True, alpha=0.3)

            diag_path = os.path.join(output_folder, "flow_qc_median_mad.png")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(diag_path, dpi=VideoDPI)
            plt.close(fig)

            print(f"[ FLOW ] QC diagnostic plot saved to: {diag_path}")
        except Exception as e:
            print(f"[ FLOW ] Warning: failed to create QC diagnostic plot: {e}")

#%%

# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    overall_start = time.perf_counter()

    # --- Input/output folders (edit these as needed) ---
    # image_folder = r"D:\Ansis Zivers\Optical Flow\Image_sequences\RBC_corrected_images_3D_filtered_TV_Gaussian_consRel_0.0175"
    # output_folder_global = r"D:\Ansis Zivers\Optical Flow\Output_updated_code"
    
    image_folder = r"D:\Work File Storage\KristofersRolands\Code\Optical_flow\Image_sequences\test_solidification_v2_filtered_3D_KLD"
    output_folder_global = r"D:\Work File Storage\KristofersRolands\Code\Optical_flow\Output_2WCF_SVD\Output_2026-01-15_extra_KLD"
    

    os.makedirs(output_folder_global, exist_ok=True)

    # Create output dirs
    output_folder = os.path.join(
        output_folder_global,
        os.path.basename(image_folder) + f"_run_{runID}"
    )
    os.makedirs(output_folder, exist_ok=True)

    output_pkl = os.path.join(output_folder, "pkl")
    os.makedirs(output_pkl, exist_ok=True)

    output_wxf = os.path.join(output_folder, "wxf")
    os.makedirs(output_wxf, exist_ok=True)

    # Save parameters (only JSON-serializable uppercase globals)
    parameters = {}
    # iterate over a static snapshot of globals() to avoid
    # "dictionary changed size during iteration" errors
    for name, value in list(globals().items()):
        if not name or not name[0].isupper():
            continue
        if name in ("In", "Out", "BytesIO", "np", "cv2", "os", "plt", "wxf",
                    "fft", "fftfreq", "imageio", "pandas", "ThreadPoolExecutor",
                    "as_completed", "tqdm"):
            continue
        # Only keep simple JSON-serializable stuff
        if isinstance(value, (int, float, bool, str)):
            parameters[name] = value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(v, (int, float, bool, str)) for v in value):
                parameters[name] = value
        elif isinstance(value, dict):
            if all(isinstance(k, str) and isinstance(v, (int, float, bool, str))
                   for k, v in value.items()):
                parameters[name] = value

    with open(os.path.join(output_folder, "parameters.json"), "w") as fp:
        json.dump(parameters, fp, indent=2)

    param_df = pandas.DataFrame(list(parameters.items()))
    param_df.to_csv(
        os.path.join(output_folder, "parameters.csv"),
        index=False,
        header=False,
    )

    print("\n[ PARAMS ] Current configuration:")
    for name, value in parameters.items():
        print(f"  {name} : {value}")

    # --- Step 1: SVD filtering (optional) ---
    if USE_SVD:
        svd_start = time.perf_counter()
        filtered_volume, image_files = svd_filter_images(
            image_folder,
            remove_modes=KillIDs,
            do_plots=False,
        )
        svd_elapsed = time.perf_counter() - svd_start
        print(f"\n[ TIMING ] SVD filtering took {svd_elapsed:.2f} s")
        images = [filtered_volume[i] for i in range(filtered_volume.shape[0])]
    else:
        svd_start = time.perf_counter()
        images, image_files = load_images_without_svd(image_folder)
        svd_elapsed = time.perf_counter() - svd_start
        print(f"\n[ TIMING ] Raw image load (no SVD) took {svd_elapsed:.2f} s")

    # --- Step 2: Optical flow ---
    flow_start = time.perf_counter()
    run_optical_flow(images, output_folder, output_pkl, output_wxf)
    flow_elapsed = time.perf_counter() - flow_start
    print(f"[ TIMING ] Optical flow + video took {flow_elapsed:.2f} s")

    overall_elapsed = time.perf_counter() - overall_start
    print(f"\n[ DONE ] Total pipeline time: {overall_elapsed:.2f} s")
