

# -*- coding: utf-8 -*-
"""
2WCF_SVD_V4_6_single_sequence_pyramid.py

Two-way consistent Farnebäck optical flow + optional temporal SVD filtering,
with timing logs, optional parallelization, and confidence-based flow smoothing
from forward/backward consistency.

Modes:
- If UseConfidenceSmoothing = True:
    * use soft confidence c = exp(-(err/σ)^2) with σ_frame = median_err + ConfidenceSigma * MAD,
      confidence-weighted local smoothing, and MinConfidence cutoff.
- If UseConfidenceSmoothing = False:
    * use a hybrid forward/backward consistency test combining:
      - a relative motion-dependent tolerance, and
      - an adaptive absolute tolerance thr_frame = median_err + HardThresholdK * MAD,
        with ConsistencyErrorThreshold as a fallback.

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

from matplotlib.colors import Normalize
from io import BytesIO
from tifffile import imread
from scipy.fft import fft, fftfreq
from scipy.ndimage import median_filter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from typing import Optional
import wolframclient.serializers as wxf

#%%

# =============================================================================
# PARAMETERS (updated from parameters.json and extended)
# =============================================================================

# --- High-level switches ---
USE_SVD = False             # If False: skip SVD and use raw TIFFs
UseParallelFlow = True      # Parallelise Farnebäck across frame pairs
MaxFlowWorkers = None       # None -> use (cpu_count - 1) if available
MaxFramePairs = None        # None -> process all frame pairs; integer -> cap the run

# --- Confidence-based smoothing parameters ---
UseConfidenceSmoothing = False  # If False: use hybrid forward/backward consistency filtering
ConfidenceSigma = 5           # k_sigma: σ_frame = median_err + k_sigma * MAD (dimensionless factor)
ConfidenceKernelSize = 5        # Neighborhood size for confidence-weighted smoothing (odd)
MinConfidence = 0.05           # Drop vectors below this confidence when sampling (soft mode)
#####################################################
UseLocalConfidenceSigma = False

# Window used to estimate local error statistics.
# Must be odd and should be larger than ConfidenceKernelSize.
LocalSigmaKernelSize = 21

# Prevent division by an extremely small local sigma.
MinimumConfidenceSigma = 1e-3
#####################################################
# --- Visualization parameters ---
PlotRejectedPoints = False
ArrowDensityStep = 6            # from parameters.json
ArrowScale = 50                 # controls arrow size inversely in quiver
VideoFPS = 15                    # from parameters.json
VideoDPI = 300                
VelocityInversion = 1          # Flip direction if vectors appear inverted
                                # Keeps the calculated velocity vectors unchanged = 1
                                # Reverses both velocity components = -1
                                # Sets every velocity vector to zero = 0
                                # make it tru/false!!!!!!!!!!!!!!!
# --- Diagnostic / validation output ---
SaveDiagnosticMaps = False
SaveValidationData = True   # Save sampled raw/processed flow + masks for quantitative validation

# --- Farneback pyramid visualization ---
# Builds a diagnostic reconstruction of the image pyramid from the SAME
# normalized uint8 frames that are passed to cv2.calcOpticalFlowFarneback.
# This is for inspection only; it does not change the optical-flow result.
SavePyramidVisualizations = False

# Frame-pair indices to visualize. [0] means only frame 0 -> 1.
# Example: [0, 10, 50]. Keep this short unless you need many diagnostics.
PyramidVisualizationPairs = [0]

# The contact sheet already shows every level. Set True only if you also want
# each level saved as a separate PNG (prev / next / absolute difference).
SavePyramidLevelImages = False

# PNG quality for the contact sheet.
PyramidVisualizationDPI = 180

# Diagnostic reconstruction stops before a level would become smaller than
# this dimension. 32 px matches the usual minimum used by OpenCV's CPU
# Farneback implementation and avoids meaningless tiny pyramid images.
PyramidVisualizationMinSize = 32

# Optional known displacement for synthetic/reference sequences.
# Set BOTH values to numbers (px/frame) to display the effective motion
# and displacement/window ratio at every pyramid level. Leave as None
# for ordinary experimental sequences where ground truth is unknown.
PyramidKnownU = 4
PyramidKnownV = 10

# Separate MP4s will be created for:
#   1. local MAD
#   2. local sigma
#   3. confidence
#
# They use the same VideoFPS and VideoDPI defined above.
                                
# --- Optical flow parameters (from parameters.json) ---
# These map directly to cv2.calcOpticalFlowFarneback(...):
#   PyramidDownsampleFactor -> pyr_scale  : image scale (<1) between pyramid levels
#   PyramidLevels           -> levels     : number of pyramid layers (including original)
#   WindowSize              -> winsize    : averaging window size; larger = smoother, more robust, but blurrier flow
#   IterationCount          -> iterations : number of iterations at each pyramid level
#   PolyFitKernelSize       -> poly_n     : size of the pixel neighborhood used for polynomial expansion
#   GaussKernelStdev        -> poly_sigma : std of Gaussian used for polynomial smoothing; larger = smoother, less detail
#   FilterFlags             -> flags      : algorithm flags (cv2.OPTFLOW_FARNEBACK_* constants)
PyramidDownsampleFactor = 0.5
PyramidLevels = 8
WindowSize = 10
IterationCount = 5
PolyFitKernelSize = 7
GaussKernelStdev = 5
FilterFlags = 0

# Fallback hard threshold for consistency; used if robust stats fail (NaNs)
ConsistencyErrorThreshold = 1.0

# Frame-adaptive absolute consistency threshold factor
# thr_frame = median_err + HardThresholdK * MAD
HardThresholdK = 3.0

# Relative forward/backward consistency term. The accepted error obeys:
# err^2 <= RelativeConsistencyAlpha * (|fwd|^2 + |bwd|^2) + thr_frame^2
RelativeConsistencyAlpha = 0.01

# Group size for median-of-means robust statistic
MedianOfMeansGroupSize = 10000


# --- Temporal SVD modes to remove (from parameters.json) ---
KillIDs = [
    0
]

# --- Run ID (careful with output overwrites) ---
runID = 1


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
        [f for f in os.listdir(image_folder) if f.lower().endswith((".tif", ".tiff"))],
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
        [f for f in os.listdir(image_folder) if f.lower().endswith((".tif", ".tiff"))],
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

def normalize_frame(frame: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    """
    Normalize frame to uint8 [0, 255] for optical flow input.
    """
    frame = frame.astype(np.float32)
    frame = frame - global_min
    value_range = global_max - global_min
    if value_range > 0:
        frame = frame / value_range
    frame = np.clip(frame, 0.0, 1.0)
    frame_uint8 = (frame * 255).astype(np.uint8)
    return frame_uint8


def build_farneback_visualization_pyramid(frame_uint8: np.ndarray):
    """
    Build a diagnostic reconstruction of the image pyramid used conceptually
    by Farneback.

    IMPORTANT
    ---------
    cv2.calcOpticalFlowFarneback does not expose its internal pyramid images.
    This function therefore reconstructs them from the same normalized uint8
    input using the configured PyramidDownsampleFactor and PyramidLevels.

    For each requested level, the image is generated directly from level 0
    after Gaussian pre-smoothing, then resized. This mirrors the structure of
    OpenCV's CPU Farneback pyramid construction closely enough for visual
    diagnostics, but it should not be described as an internal OpenCV dump.

    Returns
    -------
    levels : list[dict]
        Each dictionary contains:
          level, scale, width, height, sigma, smooth_kernel, image
    """

    frame_uint8 = np.asarray(frame_uint8)

    if frame_uint8.ndim != 2:
        raise ValueError(
            "Pyramid visualization expects one 2D grayscale frame."
        )

    if frame_uint8.dtype != np.uint8:
        frame_uint8 = np.clip(frame_uint8, 0, 255).astype(np.uint8)

    pyr_scale = float(PyramidDownsampleFactor)
    requested_levels = max(1, int(PyramidLevels))
    min_size = max(1, int(PyramidVisualizationMinSize))

    if not (0.0 < pyr_scale < 1.0):
        raise ValueError(
            "PyramidDownsampleFactor must be between 0 and 1 for "
            "pyramid visualization."
        )

    h0, w0 = frame_uint8.shape
    levels = []

    for level in range(requested_levels):
        scale = pyr_scale ** level
        width = max(1, int(round(w0 * scale)))
        height = max(1, int(round(h0 * scale)))

        # Level 0 is always valid. For coarser levels, stop before the image
        # becomes smaller than the diagnostic minimum size.
        if level > 0 and min(width, height) < min_size:
            break

        if level == 0:
            sigma = 0.0
            smooth_kernel = 1
            level_image = frame_uint8.copy()
        else:
            # OpenCV-style scale-dependent pre-smoothing before reduction.
            sigma = (1.0 / scale - 1.0) * 0.5
            smooth_kernel = max(1, int(round(sigma * 5.0)) | 1)

            if smooth_kernel > 1 and sigma > 0.0:
                smoothed = cv2.GaussianBlur(
                    frame_uint8,
                    (smooth_kernel, smooth_kernel),
                    sigmaX=sigma,
                    sigmaY=sigma,
                    borderType=cv2.BORDER_REFLECT_101,
                )
            else:
                smoothed = frame_uint8

            level_image = cv2.resize(
                smoothed,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        levels.append({
            "level": level,
            "scale": float(scale),
            "width": int(width),
            "height": int(height),
            "sigma": float(sigma),
            "smooth_kernel": int(smooth_kernel),
            "image": level_image,
        })

    return levels


def save_pyramid_visualizations(
        images,
        output_folder,
        global_min,
        global_max,
        known_u=None,
        known_v=None,
):
    """
    Save visual pyramid diagnostics for selected frame pairs.

    Output structure
    ----------------
    output_folder/
        pyramid_visualization/
            pyramid_levels.csv
            README.txt
            frame_pair_00000/
                pyramid_contact_sheet.png
                level_00_prev.png        (optional)
                level_00_next.png        (optional)
                level_00_absdiff.png     (optional)
                ...

    The contact sheet has one row per pyramid level with a metadata panel plus
    three image columns:
        metadata | previous frame | next frame | absolute difference

    If known_u/known_v are supplied (synthetic benchmark), the title for each
    level also reports the ground-truth displacement after scaling and its
    ratio to WindowSize.
    """

    if not SavePyramidVisualizations:
        return

    if len(images) < 2:
        print("[ PYRAMID ] Not enough frames for visualization.")
        return

    pyramid_root = os.path.join(
        output_folder,
        "pyramid_visualization",
    )
    os.makedirs(pyramid_root, exist_ok=True)

    # Normalize the requested indices and silently remove duplicates while
    # preserving the user's order.
    requested_pairs = []
    seen = set()
    for value in PyramidVisualizationPairs:
        pair_index = int(value)
        if pair_index not in seen:
            requested_pairs.append(pair_index)
            seen.add(pair_index)

    if not requested_pairs:
        print("[ PYRAMID ] No frame pairs selected for visualization.")
        return

    known_speed = None
    if known_u is not None and known_v is not None:
        known_speed = float(np.hypot(float(known_u), float(known_v)))

    metadata_records = []

    readme_path = os.path.join(pyramid_root, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "Farneback pyramid visualization\n"
            "================================\n\n"
            "These images are a diagnostic reconstruction of the pyramid from "
            "the same normalized uint8 frames passed to "
            "cv2.calcOpticalFlowFarneback. OpenCV does not expose its internal "
            "per-level images through this API, so these files must not be "
            "described as an exact internal pyramid dump.\n\n"
            f"PyramidDownsampleFactor = {PyramidDownsampleFactor}\n"
            f"PyramidLevels requested = {PyramidLevels}\n"
            f"WindowSize = {WindowSize}\n"
            f"PyramidVisualizationMinSize = {PyramidVisualizationMinSize}\n"
        )

    for pair_index in requested_pairs:
        if pair_index < 0 or pair_index + 1 >= len(images):
            print(
                f"[ PYRAMID ] Skipping invalid frame pair {pair_index}: "
                f"valid range is 0..{len(images) - 2}."
            )
            continue

        prev_img = normalize_frame(
            images[pair_index],
            global_min,
            global_max,
        )
        next_img = normalize_frame(
            images[pair_index + 1],
            global_min,
            global_max,
        )

        prev_levels = build_farneback_visualization_pyramid(prev_img)
        next_levels = build_farneback_visualization_pyramid(next_img)

        n_levels = min(len(prev_levels), len(next_levels))
        if n_levels == 0:
            print(
                f"[ PYRAMID ] No levels generated for frame pair {pair_index}."
            )
            continue

        pair_folder = os.path.join(
            pyramid_root,
            f"frame_pair_{pair_index:05d}",
        )
        os.makedirs(pair_folder, exist_ok=True)

        # One row per level. The first column is a compact metadata panel; the
        # remaining three columns show prev / next / absolute difference.
        # Every image panel has the same physical size, so even a 32x32 coarse
        # level remains easy to inspect.
        fig, axes = plt.subplots(
            n_levels,
            4,
            figsize=(14, max(3.0, 3.0 * n_levels)),
            squeeze=False,
            gridspec_kw={
                "width_ratios": [1.15, 2.0, 2.0, 2.0],
            },
        )

        for level_idx in range(n_levels):
            prev_level = prev_levels[level_idx]
            next_level = next_levels[level_idx]

            prev_level_img = prev_level["image"]
            next_level_img = next_level["image"]
            absdiff = cv2.absdiff(prev_level_img, next_level_img)

            level = prev_level["level"]
            scale = prev_level["scale"]
            width = prev_level["width"]
            height = prev_level["height"]
            sigma = prev_level["sigma"]
            smooth_kernel = prev_level["smooth_kernel"]

            effective_displacement = float("nan")
            displacement_window_ratio = float("nan")
            if known_speed is not None:
                effective_displacement = known_speed * scale
                if float(WindowSize) > 0:
                    displacement_window_ratio = (
                        effective_displacement / float(WindowSize)
                    )

            metadata_lines = [
                f"LEVEL {level}",
                f"{width} x {height} px",
                f"scale = {scale:.4g}",
                f"sigma = {sigma:.3g}",
                f"kernel = {smooth_kernel}",
            ]
            if known_speed is not None:
                metadata_lines.extend([
                    f"GT d = {effective_displacement:.3g} px",
                    f"d / W = {displacement_window_ratio:.3g}",
                ])

            axes[level_idx, 0].axis("off")
            axes[level_idx, 0].text(
                0.5,
                0.5,
                "\n".join(metadata_lines),
                transform=axes[level_idx, 0].transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )

            axes[level_idx, 1].imshow(
                prev_level_img,
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
            axes[level_idx, 2].imshow(
                next_level_img,
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
            axes[level_idx, 3].imshow(
                absdiff,
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )

            for axis in axes[level_idx, 1:]:
                axis.set_xticks([])
                axis.set_yticks([])

            if SavePyramidLevelImages:
                imageio.imwrite(
                    os.path.join(
                        pair_folder,
                        f"level_{level:02d}_prev.png",
                    ),
                    prev_level_img,
                )
                imageio.imwrite(
                    os.path.join(
                        pair_folder,
                        f"level_{level:02d}_next.png",
                    ),
                    next_level_img,
                )
                imageio.imwrite(
                    os.path.join(
                        pair_folder,
                        f"level_{level:02d}_absdiff.png",
                    ),
                    absdiff,
                )

            metadata_records.append({
                "frame_pair": pair_index,
                "level": level,
                "scale": scale,
                "width_px": width,
                "height_px": height,
                "gaussian_sigma_px_at_level0": sigma,
                "gaussian_kernel_px_at_level0": smooth_kernel,
                "known_u_px_per_frame": (
                    float(known_u) if known_u is not None else float("nan")
                ),
                "known_v_px_per_frame": (
                    float(known_v) if known_v is not None else float("nan")
                ),
                "known_speed_px_per_frame": (
                    known_speed if known_speed is not None else float("nan")
                ),
                "effective_displacement_px": effective_displacement,
                "window_size_px": float(WindowSize),
                "displacement_to_window_ratio": displacement_window_ratio,
            })

        axes[0, 0].set_title("Level metadata", fontsize=11)
        axes[0, 1].set_title("Previous frame", fontsize=11)
        axes[0, 2].set_title("Next frame", fontsize=11)
        axes[0, 3].set_title("Absolute difference", fontsize=11)

        fig.suptitle(
            "Farneback diagnostic image pyramid\n"
            f"Frame {pair_index} -> {pair_index + 1} | "
            f"pyr_scale={PyramidDownsampleFactor} | "
            f"requested levels={PyramidLevels} | "
            f"visualized levels={n_levels} | winsize={WindowSize}",
            fontsize=12,
        )

        fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.95])

        contact_sheet_path = os.path.join(
            pair_folder,
            "pyramid_contact_sheet.png",
        )
        fig.savefig(
            contact_sheet_path,
            dpi=PyramidVisualizationDPI,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(
            f"[ PYRAMID ] Saved frame pair {pair_index} -> {pair_index + 1}: "
            f"{n_levels} levels to {contact_sheet_path}"
        )

        del prev_img, next_img, prev_levels, next_levels

    if metadata_records:
        metadata_df = pandas.DataFrame(metadata_records)
        metadata_path = os.path.join(
            pyramid_root,
            "pyramid_levels.csv",
        )
        metadata_df.to_csv(metadata_path, index=False)
        print(f"[ PYRAMID ] Metadata saved: {metadata_path}")


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
#%%new shit
def calculate_local_sigma(consistency_error: np.ndarray) -> tuple:
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
    k = LocalSigmaKernelSize

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
        + ConfidenceSigma * local_mad
    ).astype(np.float32)

    # Prevent zero or extremely small sigma because it is later used as a divisor
    sigma_map = np.maximum(
        sigma_map,
        MinimumConfidenceSigma
    ).astype(np.float32)

    return sigma_map, local_mad
#%%
def confidence_weighted_smoothing(
        u: np.ndarray,
        v: np.ndarray,
        consistency_error: np.ndarray,
        sigma: float,
        reliability_mask: Optional[np.ndarray] = None
) -> tuple:
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
    & (sigma_safe > MinimumConfidenceSigma),
    sigma_safe,
    MinimumConfidenceSigma)

   # Confidence in [0, 1]
    conf = np.exp(-(consistency_error / sigma_safe) ** 2).astype(np.float32)
    
    # Completely prevent inconsistent vectors from contributing
    # to the confidence-weighted neighbourhood average
    if reliability_mask is not None:
        conf *= reliability_mask.astype(np.float32)
    
    k = ConfidenceKernelSize
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


def compute_flow_pair(index: int, images, h: int, w: int, global_min, global_max):
    """
    Compute two-way-consistent Farneback flow for frame pair (index, index+1).

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
        mom = median_of_means(err_flat, group_size=MedianOfMeansGroupSize)
    else:
        med = float("nan")
        mad = float("nan")
        mom = float("nan")

    if np.isfinite(med) and np.isfinite(mad):
        thr_frame = med + HardThresholdK * mad
        sigma_frame = med + ConfidenceSigma * mad
    else:
        thr_frame = ConsistencyErrorThreshold
        sigma_frame = 1.0

    if not np.isfinite(thr_frame):
        thr_frame = ConsistencyErrorThreshold
    if not np.isfinite(sigma_frame) or sigma_frame <= 0:
        sigma_frame = MinimumConfidenceSigma

    # Sample grid
    sample_y = slice(ArrowDensityStep // 2, h, ArrowDensityStep)
    sample_x = slice(ArrowDensityStep // 2, w, ArrowDensityStep)

    y_grid, x_grid = np.mgrid[
        ArrowDensityStep // 2:h:ArrowDensityStep,
        ArrowDensityStep // 2:w:ArrowDensityStep,
    ]

    # -----------------------------------------------------------------
    # Hybrid forward/backward consistency test
    # -----------------------------------------------------------------
    forward_magnitude_sq = u**2 + v**2
    backward_magnitude_sq = bwd_u_warped**2 + bwd_v_warped**2

    allowed_error_squared = (
        RelativeConsistencyAlpha
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
    need_local_stats = UseLocalConfidenceSigma or SaveDiagnosticMaps
    local_sigma_map = None
    local_mad_map = None

    if need_local_stats:
        local_sigma_map, local_mad_map = calculate_local_sigma(
            consistency_error
        )

    confidence_map_for_diagnostics = None

    if UseConfidenceSmoothing:
        # Soft mode genuinely needs a dense confidence field because the
        # confidence-weighted box filters operate on every pixel.
        sigma_for_confidence = (
            local_sigma_map if UseLocalConfidenceSigma else sigma_frame
        )

        u_s, v_s, conf_map = confidence_weighted_smoothing(
            u,
            v,
            consistency_error,
            sigma_for_confidence,
            reliability_mask,
        )

        u_sampled = u_s[sample_y, sample_x]
        v_sampled = v_s[sample_y, sample_x]
        conf_sampled = conf_map[sample_y, sample_x]

        valid_indices = (
            mask_sampled
            & (conf_sampled >= MinConfidence)
        )

        if SaveDiagnosticMaps:
            confidence_map_for_diagnostics = conf_map

    else:
        # Hard mode does not need a dense confidence map unless diagnostic
        # map output was explicitly requested. For validation, calculate
        # confidence only on the sampled grid.
        u_sampled = u_raw_sampled
        v_sampled = v_raw_sampled

        if UseLocalConfidenceSigma:
            sigma_sampled = local_sigma_map[sample_y, sample_x]
        else:
            sigma_sampled = np.float32(sigma_frame)

        sigma_sampled = np.asarray(sigma_sampled, dtype=np.float32)
        sigma_sampled = np.where(
            np.isfinite(sigma_sampled)
            & (sigma_sampled > MinimumConfidenceSigma),
            sigma_sampled,
            MinimumConfidenceSigma,
        )

        conf_sampled = np.exp(
            -(consistency_error_sampled / sigma_sampled) ** 2
        ).astype(np.float32)

        valid_indices = mask_sampled

        if SaveDiagnosticMaps:
            if UseLocalConfidenceSigma:
                sigma_full = local_sigma_map
            else:
                sigma_full = np.float32(sigma_frame)

            sigma_full = np.asarray(sigma_full, dtype=np.float32)
            sigma_full = np.where(
                np.isfinite(sigma_full)
                & (sigma_full > MinimumConfidenceSigma),
                sigma_full,
                MinimumConfidenceSigma,
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

    if UseConfidenceSmoothing:
        low_confidence_vectors = int(np.count_nonzero(
            mask_sampled & (conf_sampled < MinConfidence)
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
        f"relative alpha={RelativeConsistencyAlpha:.4g}, "
        f"effective limit median={effective_limit_median:.4f} px "
        f"(p10={effective_limit_p10:.4f}, p90={effective_limit_p90:.4f}), "
        f"median error={med:.4f}, MAD={mad:.4f}"
    )

    u_output_sampled = VelocityInversion * u_sampled
    v_output_sampled = VelocityInversion * v_sampled

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
        "relative_consistency_alpha": float(RelativeConsistencyAlpha),
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
    if SaveDiagnosticMaps:
        result["local_mad_map"] = local_mad_map.astype(np.float32, copy=False)
        result["local_sigma_map"] = local_sigma_map.astype(np.float32, copy=False)
        result["confidence_map"] = confidence_map_for_diagnostics.astype(
            np.float32, copy=False
        )

    return result

def write_scalar_map_video(
        diagnostic_paths,
        map_key,
        output_path,
        title,
        colorbar_label,
        vmin,
        vmax
):
    """
    Create an MP4 video from scalar diagnostic maps saved in NPZ files.

    All frames use the SAME vmin/vmax, so a given colour represents
    the same numerical value throughout the whole video.
    """

    print(f"\n[ DIAGNOSTIC VIDEO ] Creating: {output_path}")

    # Safety against a completely constant map
    if not np.isfinite(vmin):
        vmin = 0.0

    if not np.isfinite(vmax):
        vmax = 1.0

    if vmax <= vmin:
        vmax = vmin + 1e-6

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = None
    frame_size = None

    for frame_index, diagnostic_path in tqdm(
            list(enumerate(diagnostic_paths)),
            desc=f"Writing {title}"
    ):

        # ---------------------------------------------------------
        # Load raw map
        # ---------------------------------------------------------
        with np.load(diagnostic_path) as diagnostic_data:
            scalar_map = diagnostic_data[map_key].astype(np.float32)

        # ---------------------------------------------------------
        # Plot map
        # ---------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(7, 6),
            dpi=VideoDPI
        )

        image = ax.imshow(
            scalar_map,
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        colorbar.set_label(colorbar_label)

        ax.set_title(
            f"{title}\n"
            f"Frame {frame_index} → {frame_index + 1}"
        )

        ax.axis("off")

        fig.tight_layout()

        # ---------------------------------------------------------
        # Convert Matplotlib figure into video frame
        # ---------------------------------------------------------
        buf = BytesIO()

        fig.savefig(
            buf,
            format="png"
        )

        plt.close(fig)

        buf.seek(0)

        frame = imageio.imread(buf)
        buf.close()

        if frame.ndim == 2:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2RGB
            )

        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # ---------------------------------------------------------
        # Initialise MP4 writer
        # ---------------------------------------------------------
        if video_writer is None:

            frame_size = (
                frame.shape[1],
                frame.shape[0]
            )

            video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                VideoFPS,
                frame_size
            )

            if not video_writer.isOpened():
                raise IOError(
                    f"Failed to open video writer: {output_path}"
                )

        # Matplotlib gives RGB; OpenCV expects BGR
        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        frame_bgr = cv2.resize(
            frame_bgr,
            frame_size
        )

        video_writer.write(frame_bgr)

    if video_writer is not None:
        video_writer.release()

    print(
        f"[ DIAGNOSTIC VIDEO ] Saved: {output_path}"
    )

def run_optical_flow(images,
                     output_folder,
                     output_pkl,
                     output_wxf,
                     global_min,
                     global_max):
    """
    Run two-way-consistent Farneback optical flow with streaming disk output.

    MEMORY DESIGN
    -------------
    A frame-pair result is consumed as soon as that Future finishes:

        compute -> save PKL/WXF/validation/diagnostics -> keep tiny QC scalars
                -> release the result -> submit another pair

    There is no all-frame ``results`` dictionary. Therefore completed frame
    pairs do not accumulate in RAM. The number of large in-flight results is
    bounded by ``workers``.

    The velocity video is intentionally a second disk-backed pass. During the
    first pass we update the global speed range while each result is still in
    memory, then discard it. During the second pass we read one PKL at a time
    to render the video with a fixed global colour scale.
    """
    print("\n[ FLOW ] Starting optical flow analysis...")

    if len(images) < 2:
        print("[ FLOW ] Not enough frames for optical flow.")
        return

    h, w = images[0].shape
    if MaxFramePairs is None:
        indices = list(range(len(images) - 1))
    else:
        indices = list(range(min(len(images) - 1, int(MaxFramePairs))))

    if not indices:
        print("[ FLOW ] No frame pairs selected.")
        return

    # -------------------------------------------------------------------------
    # Output folders
    # -------------------------------------------------------------------------
    diagnostic_folder = os.path.join(output_folder, "diagnostic_maps")
    validation_folder = os.path.join(output_folder, "validation")

    if SaveDiagnosticMaps:
        os.makedirs(diagnostic_folder, exist_ok=True)

    if SaveValidationData:
        os.makedirs(validation_folder, exist_ok=True)

    # Rejected points are needed only for the optional video overlay. They are
    # cached on disk rather than held for all frames in memory.
    rejected_cache_folder = os.path.join(output_folder, "_video_rejected_cache")
    rejected_cache_paths = {}
    if PlotRejectedPoints:
        os.makedirs(rejected_cache_folder, exist_ok=True)

    # -------------------------------------------------------------------------
    # Tiny run-level state: scalars, paths and QC records only.
    # -------------------------------------------------------------------------
    qc_records = []
    diagnostic_paths_by_index = {}

    diagnostic_ranges = {
        "local_mad": [np.inf, -np.inf],
        "local_sigma": [np.inf, -np.inf],
    }

    global_speed_min = np.inf
    global_speed_max = -np.inf

    def update_diagnostic_range(name, array):
        finite_values = array[np.isfinite(array)]
        if finite_values.size == 0:
            return

        current_min = float(np.min(finite_values))
        current_max = float(np.max(finite_values))

        diagnostic_ranges[name][0] = min(
            diagnostic_ranges[name][0], current_min
        )
        diagnostic_ranges[name][1] = max(
            diagnostic_ranges[name][1], current_max
        )

    def save_diagnostics(res):
        """Write full-resolution maps immediately and remove them from res."""
        index = res["index"]

        local_mad_map = res.pop("local_mad_map")
        local_sigma_map = res.pop("local_sigma_map")
        confidence_map = res.pop("confidence_map")

        update_diagnostic_range("local_mad", local_mad_map)
        update_diagnostic_range("local_sigma", local_sigma_map)

        diagnostic_path = os.path.join(
            diagnostic_folder,
            f"diagnostic_maps_{index:05d}.npz",
        )

        np.savez_compressed(
            diagnostic_path,
            local_mad=local_mad_map,
            local_sigma=local_sigma_map,
            confidence=confidence_map,
        )

        diagnostic_paths_by_index[index] = diagnostic_path

        # Drop explicit references before continuing with other disk writes.
        del local_mad_map, local_sigma_map, confidence_map

    def save_validation_data(res):
        """Write sampled validation arrays immediately and remove them from res."""
        index = res["index"]
        validation_path = os.path.join(
            validation_folder,
            f"validation_{index:05d}.npz",
        )

        validation_keys = [
            "validation_x_all", "validation_y_all",
            "validation_u_raw", "validation_v_raw",
            "validation_u_processed", "validation_v_processed",
            "validation_u_output", "validation_v_output",
            "validation_valid_mask", "validation_consistency_mask",
            "validation_consistency_error", "validation_confidence",
        ]

        np.savez_compressed(
            validation_path,
            x=res["validation_x_all"],
            y=res["validation_y_all"],
            u_raw=res["validation_u_raw"],
            v_raw=res["validation_v_raw"],
            u_processed=res["validation_u_processed"],
            v_processed=res["validation_v_processed"],
            u_output=res["validation_u_output"],
            v_output=res["validation_v_output"],
            valid_mask=res["validation_valid_mask"],
            consistency_mask=res["validation_consistency_mask"],
            consistency_error=res["validation_consistency_error"],
            confidence=res["validation_confidence"],
        )

        for key in validation_keys:
            res.pop(key, None)

    def save_velocity_field(res):
        """Write the normal final retained field immediately."""
        index = res["index"]

        flat_data = {
            "x": np.asarray(res["x"]).ravel(),
            "y": np.asarray(res["y"]).ravel(),
            "u": np.asarray(res["u"]).ravel(),
            "v": np.asarray(res["v"]).ravel(),
        }

        pkl_path = os.path.join(
            output_pkl,
            f"velocity_field_{index:05d}.pkl",
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(flat_data, f, pickle.HIGHEST_PROTOCOL)

        wxf_path = os.path.join(
            output_wxf,
            f"velocity_field_{index:05d}.wxf",
        )
        wxf.export(
            flat_data,
            wxf_path,
            target_format="wxf",
        )

    def cache_rejected_points_for_video(res):
        if not PlotRejectedPoints:
            return

        index = res["index"]
        cache_path = os.path.join(
            rejected_cache_folder,
            f"rejected_{index:05d}.npz",
        )
        np.savez_compressed(
            cache_path,
            x_rejected=res["x_rejected"],
            y_rejected=res["y_rejected"],
        )
        rejected_cache_paths[index] = cache_path

    def collect_qc_record(res):
        qc_records.append({
            "frame_index": res["index"],
            "median_err": res.get("median_err", float("nan")),
            "mad_err": res.get("mad_err", float("nan")),
            "mom_err": res.get("mom_err", float("nan")),
            "thr_frame": res.get("thr_frame", float("nan")),
            "relative_consistency_alpha": res.get(
                "relative_consistency_alpha", float("nan")
            ),
            "effective_limit_median": res.get(
                "effective_limit_median", float("nan")
            ),
            "effective_limit_p10": res.get(
                "effective_limit_p10", float("nan")
            ),
            "effective_limit_p90": res.get(
                "effective_limit_p90", float("nan")
            ),
            "sigma_frame": res.get("sigma_frame", float("nan")),
            "total_vectors": res.get("total_vectors", 0),
            "retained_vectors": res.get("retained_vectors", 0),
            "rejected_vectors": res.get("rejected_vectors", 0),
            "consistency_rejected_vectors": res.get(
                "consistency_rejected_vectors", 0
            ),
            "low_confidence_vectors": res.get("low_confidence_vectors", 0),
            "retention_fraction": res.get(
                "retention_fraction", float("nan")
            ),
            "rejection_fraction": res.get(
                "rejection_fraction", float("nan")
            ),
            "use_confidence_smoothing": bool(UseConfidenceSmoothing),
        })

    def consume_result(res):
        """
        Persist one completed frame pair immediately.

        After this returns, the caller can delete ``res``. No frame-pair-sized
        object is needed later because the video is rendered from disk.
        """
        nonlocal global_speed_min, global_speed_max

        # Update global speed limits before discarding the vectors.
        speed = np.hypot(
            np.asarray(res["u"], dtype=float),
            np.asarray(res["v"], dtype=float),
        )
        finite_speed = speed[np.isfinite(speed)]
        if finite_speed.size > 0:
            global_speed_min = min(
                global_speed_min,
                float(np.min(finite_speed)),
            )
            global_speed_max = max(
                global_speed_max,
                float(np.max(finite_speed)),
            )
        del speed, finite_speed

        if SaveDiagnosticMaps:
            save_diagnostics(res)

        if SaveValidationData:
            save_validation_data(res)
        else:
            # Even when validation saving is disabled, remove the sampled
            # validation payload because it has no later use.
            for key in [
                "validation_x_all", "validation_y_all",
                "validation_u_raw", "validation_v_raw",
                "validation_u_processed", "validation_v_processed",
                "validation_u_output", "validation_v_output",
                "validation_valid_mask", "validation_consistency_mask",
                "validation_consistency_error", "validation_confidence",
            ]:
                res.pop(key, None)

        save_velocity_field(res)
        cache_rejected_points_for_video(res)
        collect_qc_record(res)

    # =========================================================================
    # PASS 1 — COMPUTE + IMMEDIATE DISK WRITE
    # =========================================================================
    start_flow = time.perf_counter()

    if UseParallelFlow and len(indices) > 1:
        cpu_count = os.cpu_count() or 1
        workers = MaxFlowWorkers or max(1, cpu_count - 1)
        workers = min(workers, len(indices))

        print(
            f"[ FLOW ] Running Farneback in parallel with {workers} workers..."
        )
        print(
            "[ FLOW ] Streaming mode: each completed frame pair is written "
            "to disk immediately."
        )

        # BOUNDED in-flight queue. We do NOT submit all frame pairs up front.
        # At most `workers` Futures can own large results at any one time.
        index_iterator = iter(indices)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            in_flight = {}

            # Prime the worker pool.
            for _ in range(workers):
                try:
                    i = next(index_iterator)
                except StopIteration:
                    break

                future = executor.submit(
                    compute_flow_pair,
                    i,
                    images,
                    h,
                    w,
                    global_min,
                    global_max,
                )
                in_flight[future] = i

            with tqdm(
                total=len(indices),
                desc="Farneback + immediate disk write",
            ) as progress_bar:

                while in_flight:
                    done, _ = wait(
                        in_flight,
                        return_when=FIRST_COMPLETED,
                    )

                    for future in done:
                        i = in_flight.pop(future)
                        res = future.result()

                        try:
                            consume_result(res)
                        finally:
                            # Drop the main-thread reference immediately.
                            del res

                        progress_bar.update(1)

                        # Refill exactly one slot. This keeps the number of
                        # large in-flight computations bounded by `workers`.
                        try:
                            next_i = next(index_iterator)
                        except StopIteration:
                            continue

                        next_future = executor.submit(
                            compute_flow_pair,
                            next_i,
                            images,
                            h,
                            w,
                            global_min,
                            global_max,
                        )
                        in_flight[next_future] = next_i

    else:
        print("[ FLOW ] Running Farneback sequentially in streaming mode...")

        for i in tqdm(
            indices,
            desc="Farneback + immediate disk write",
        ):
            res = compute_flow_pair(
                i,
                images,
                h,
                w,
                global_min,
                global_max,
            )
            try:
                consume_result(res)
            finally:
                del res

    flow_elapsed = time.perf_counter() - start_flow
    avg_per_pair = flow_elapsed / max(1, len(indices))
    print(
        f"[ FLOW ] Flow computation + immediate writes: "
        f"{flow_elapsed:.2f} s total ({avg_per_pair:.2f} s per frame pair)."
    )

    # Sort tiny QC records because parallel Futures finish out of order.
    qc_records.sort(key=lambda record: record["frame_index"])

    # =========================================================================
    # GLOBAL VELOCITY COLOR SCALE
    # =========================================================================
    if (
        not np.isfinite(global_speed_min)
        or not np.isfinite(global_speed_max)
    ):
        global_speed_min = 0.0
        global_speed_max = 1.0

    if global_speed_max <= global_speed_min:
        epsilon = max(abs(global_speed_min) * 1e-6, 1e-6)
        global_speed_min -= epsilon
        global_speed_max += epsilon

    velocity_color_norm = Normalize(
        vmin=global_speed_min,
        vmax=global_speed_max,
    )

    print("\n[ VELOCITY COLOR SCALE ]")
    print(
        f"  Global minimum speed: {global_speed_min:.6g} px/frame"
    )
    print(
        f"  Global maximum speed: {global_speed_max:.6g} px/frame"
    )

    # =========================================================================
    # PASS 2 — VIDEO FROM DISK, ONE FRAME AT A TIME
    # =========================================================================
    video_path = os.path.join(output_folder, "velocity_field.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None
    frame_size = None

    print(
        "[ FLOW ] Building velocity video from disk-backed PKL files..."
    )

    for i in tqdm(indices, desc="Velocity video from disk"):
        pkl_path = os.path.join(
            output_pkl,
            f"velocity_field_{i:05d}.pkl",
        )

        with open(pkl_path, "rb") as f:
            flat_data = pickle.load(f)

        x = np.asarray(flat_data["x"], dtype=float)
        y = np.asarray(flat_data["y"], dtype=float)
        u_vec = np.asarray(flat_data["u"], dtype=float)
        v_vec = np.asarray(flat_data["v"], dtype=float)

        if PlotRejectedPoints:
            cache_path = rejected_cache_paths.get(i)
            if cache_path and os.path.exists(cache_path):
                with np.load(cache_path) as rejected_data:
                    x_rejected = rejected_data["x_rejected"]
                    y_rejected = rejected_data["y_rejected"]
            else:
                x_rejected = np.array([], dtype=float)
                y_rejected = np.array([], dtype=float)
        else:
            x_rejected = np.array([], dtype=float)
            y_rejected = np.array([], dtype=float)

        prev_img = normalize_frame(images[i], global_min, global_max)

        fig, ax = plt.subplots(
            figsize=(6, 6),
            dpi=VideoDPI,
        )
        ax.imshow(prev_img, cmap="gray")

        speed = np.hypot(u_vec, v_vec)
        finite = np.isfinite(speed)

        normalisation_mask = (
            finite
            & (speed > np.finfo(float).eps)
        )

        u_norm = np.divide(
            u_vec,
            speed,
            out=np.zeros_like(u_vec, dtype=float),
            where=normalisation_mask,
        )
        v_norm = np.divide(
            v_vec,
            speed,
            out=np.zeros_like(v_vec, dtype=float),
            where=normalisation_mask,
        )

        quiver = ax.quiver(
            x,
            y,
            u_norm,
            v_norm,
            speed,
            cmap="jet",
            norm=velocity_color_norm,
            scale=ArrowScale,
            angles="xy",
            zorder=3,
        )

        colorbar = fig.colorbar(
            quiver,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("Velocity magnitude [px/frame]")

        if PlotRejectedPoints and x_rejected.size > 0:
            ax.scatter(
                x_rejected,
                y_rejected,
                s=4,
                c="magenta",
                marker=".",
                alpha=0.9,
                zorder=4,
                label="Rejected by final validity test",
            )

        ax.set_title(f"Frame {i} → {i + 1}")
        ax.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        video_frame = imageio.imread(buf)
        buf.close()

        if video_frame.ndim == 2:
            video_frame = cv2.cvtColor(
                video_frame,
                cv2.COLOR_GRAY2RGB,
            )
        elif video_frame.shape[2] == 4:
            video_frame = video_frame[:, :, :3]

        if video_writer is None:
            frame_size = (
                video_frame.shape[1],
                video_frame.shape[0],
            )
            video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                VideoFPS,
                frame_size,
            )
            if not video_writer.isOpened():
                raise IOError(
                    f"Failed to open video writer at: {video_path}"
                )

        frame_bgr = cv2.cvtColor(
            video_frame,
            cv2.COLOR_RGB2BGR,
        )
        resized = cv2.resize(frame_bgr, frame_size)
        video_writer.write(resized)

        # Explicitly release per-video-frame arrays before the next PKL.
        del (
            flat_data, x, y, u_vec, v_vec, speed, finite,
            normalisation_mask, u_norm, v_norm, video_frame,
            frame_bgr, resized, x_rejected, y_rejected,
        )

    if video_writer is not None:
        video_writer.release()

    print(f"[ FLOW ] Optical flow MP4 saved to: {video_path}")

    # Remove the optional temporary rejected-point cache after the video exists.
    if PlotRejectedPoints:
        for cache_path in rejected_cache_paths.values():
            try:
                os.remove(cache_path)
            except OSError:
                pass
        try:
            os.rmdir(rejected_cache_folder)
        except OSError:
            pass

    # =========================================================================
    # DIAGNOSTIC VIDEOS — already disk-backed
    # =========================================================================
    if SaveDiagnosticMaps:
        print("\n[ DIAGNOSTICS ] Creating diagnostic map videos...")

        diagnostic_paths = [
            diagnostic_paths_by_index[i]
            for i in indices
        ]

        mad_vmin, mad_vmax = diagnostic_ranges["local_mad"]
        sigma_vmin, sigma_vmax = diagnostic_ranges["local_sigma"]
        confidence_vmin = 0.0
        confidence_vmax = 1.0

        print(
            f"[ DIAGNOSTICS ] Local MAD scale: "
            f"{mad_vmin:.6g} → {mad_vmax:.6g}"
        )
        print(
            f"[ DIAGNOSTICS ] Local sigma scale: "
            f"{sigma_vmin:.6g} → {sigma_vmax:.6g}"
        )
        print("[ DIAGNOSTICS ] Confidence scale: 0 → 1")

        write_scalar_map_video(
            diagnostic_paths=diagnostic_paths,
            map_key="local_mad",
            output_path=os.path.join(output_folder, "local_mad.mp4"),
            title="Local Median Absolute Deviation",
            colorbar_label="Local scaled MAD",
            vmin=mad_vmin,
            vmax=mad_vmax,
        )

        write_scalar_map_video(
            diagnostic_paths=diagnostic_paths,
            map_key="local_sigma",
            output_path=os.path.join(output_folder, "local_sigma.mp4"),
            title="Local Confidence Sigma",
            colorbar_label="Local sigma",
            vmin=sigma_vmin,
            vmax=sigma_vmax,
        )

        write_scalar_map_video(
            diagnostic_paths=diagnostic_paths,
            map_key="confidence",
            output_path=os.path.join(output_folder, "confidence.mp4"),
            title="Optical Flow Confidence",
            colorbar_label="Confidence",
            vmin=confidence_vmin,
            vmax=confidence_vmax,
        )

        print("\n[ DIAGNOSTICS ] Finished diagnostic videos.")

    # -------------------------------------------------------------------------
    # QC CSV
    # -------------------------------------------------------------------------
    if qc_records:
        qc_df = pandas.DataFrame(qc_records)
        qc_csv_path = os.path.join(
            output_folder,
            "flow_qc_metrics.csv",
        )
        qc_df.to_csv(qc_csv_path, index=False)
        print(f"[ FLOW ] QC metrics saved to: {qc_csv_path}")

    # -------------------------------------------------------------------------
    # QC plot
    # -------------------------------------------------------------------------
    if qc_records:
        try:
            frames = [rec["frame_index"] for rec in qc_records]
            medians = [rec["median_err"] for rec in qc_records]
            mads = [rec["mad_err"] for rec in qc_records]

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 6), sharex=True
            )
            fig.suptitle("Forward–Backward Consistency Diagnostics")

            ax1.plot(frames, medians, marker="o")
            ax1.set_ylabel("median_err")
            ax1.grid(True, alpha=0.3)

            ax2.plot(frames, mads, marker="o")
            ax2.set_xlabel("frame index")
            ax2.set_ylabel("mad_err")
            ax2.grid(True, alpha=0.3)

            diag_path = os.path.join(
                output_folder,
                "flow_qc_median_mad.png",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(diag_path, dpi=VideoDPI)
            plt.close(fig)

            print(
                f"[ FLOW ] QC diagnostic plot saved to: {diag_path}"
            )
        except Exception as e:
            print(
                "[ FLOW ] Warning: failed to create QC diagnostic plot: "
                f"{e}"
            )

#%%

# =============================================================================
# MAIN PIPELINE — ONE IMAGE SEQUENCE
# =============================================================================

if __name__ == "__main__":
    overall_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # INPUT / OUTPUT
    # -------------------------------------------------------------------------
    # Point image_folder directly to ONE directory containing the TIFF sequence.
    # The script processes only this sequence; there is no sweep/batch discovery.
    image_folder = r"E:/Ansis Zivers/Optical Flow/Image_sequences/SYNTHETIC/Uniform_flow/Sweep/synthetic_uniform_u4_v10/frames"
    output_folder_global = r"E:/Ansis Zivers/Optical Flow/Output_2WCF_SVD/SYNTHETIC/Uniform_flow"

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            "Input image folder does not exist:\n"
            f"{image_folder}"
        )

    os.makedirs(output_folder_global, exist_ok=True)

    # If the selected directory is literally called "frames" (common for the
    # synthetic tests), use the parent dataset name so output folders do not all
    # become "frames_run_X".
    normalized_input = os.path.normpath(image_folder)
    sequence_name = os.path.basename(normalized_input)
    if sequence_name.lower() == "frames":
        parent_name = os.path.basename(os.path.dirname(normalized_input))
        if parent_name:
            sequence_name = parent_name

    output_folder = os.path.join(
        output_folder_global,
        sequence_name + f"_run_{runID}",
    )
    os.makedirs(output_folder, exist_ok=True)

    output_pkl = os.path.join(output_folder, "pkl")
    output_wxf = os.path.join(output_folder, "wxf")
    os.makedirs(output_pkl, exist_ok=True)
    os.makedirs(output_wxf, exist_ok=True)

    # -------------------------------------------------------------------------
    # SAVE CURRENT PARAMETERS
    # -------------------------------------------------------------------------
    parameters = {}

    # Iterate over a static snapshot of globals() so the dictionary cannot
    # change while it is being inspected.
    for name, value in list(globals().items()):
        if not name or not name[0].isupper():
            continue

        if name in (
            "In", "Out", "BytesIO", "np", "cv2", "os", "plt", "wxf",
            "fft", "fftfreq", "imageio", "pandas", "ThreadPoolExecutor",
            "wait", "FIRST_COMPLETED", "tqdm",
        ):
            continue

        if value is None or isinstance(value, (int, float, bool, str)):
            parameters[name] = value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(v, (int, float, bool, str)) for v in value):
                parameters[name] = value
        elif isinstance(value, dict):
            if all(
                isinstance(k, str)
                and isinstance(v, (int, float, bool, str))
                for k, v in value.items()
            ):
                parameters[name] = value

    with open(
        os.path.join(output_folder, "parameters.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(parameters, fp, indent=2)

    param_df = pandas.DataFrame(list(parameters.items()))
    param_df.to_csv(
        os.path.join(output_folder, "parameters.csv"),
        index=False,
        header=False,
    )

    print("\n============================================================")
    print("2WCF SINGLE-SEQUENCE RUN")
    print("============================================================")
    print(f"Input:  {image_folder}")
    print(f"Output: {output_folder}")
    print(f"Parallel Farneback: {UseParallelFlow}")
    print(f"Max workers:        {MaxFlowWorkers}")
    print(f"Frame-pair cap:     {MaxFramePairs}")
    print("============================================================")

    print("\n[ PARAMS ] Current configuration:")
    for name, value in parameters.items():
        print(f"  {name} : {value}")

    # -------------------------------------------------------------------------
    # STEP 1 — LOAD / OPTIONAL SVD FILTERING
    # -------------------------------------------------------------------------
    if USE_SVD:
        svd_start = time.perf_counter()
        filtered_volume, image_files = svd_filter_images(
            image_folder,
            remove_modes=KillIDs,
            do_plots=False,
        )
        svd_elapsed = time.perf_counter() - svd_start
        print(f"\n[ TIMING ] SVD filtering took {svd_elapsed:.2f} s")

        # Individual slices are views into filtered_volume, so this does not
        # duplicate the full filtered stack in RAM.
        images = [
            filtered_volume[i]
            for i in range(filtered_volume.shape[0])
        ]
    else:
        load_start = time.perf_counter()
        images, image_files = load_images_without_svd(image_folder)
        load_elapsed = time.perf_counter() - load_start
        print(f"\n[ TIMING ] Raw image load (no SVD) took {load_elapsed:.2f} s")

    # Calculate ONE global intensity range for the actual image sequence passed
    # to Farneback. This preserves the existing normalization behaviour.
    global_min = min(np.min(img) for img in images)
    global_max = max(np.max(img) for img in images)

    print(f"Global min: {global_min}")
    print(f"Global max: {global_max}")

    # -------------------------------------------------------------------------
    # STEP 2 — PYRAMID VISUALIZATION (OPTIONAL)
    # -------------------------------------------------------------------------
    if SavePyramidVisualizations:
        pyramid_start = time.perf_counter()

        save_pyramid_visualizations(
            images=images,
            output_folder=output_folder,
            global_min=global_min,
            global_max=global_max,
            known_u=PyramidKnownU,
            known_v=PyramidKnownV,
        )

        pyramid_elapsed = time.perf_counter() - pyramid_start
        print(
            f"[ TIMING ] Pyramid visualization took "
            f"{pyramid_elapsed:.2f} s"
        )

    # -------------------------------------------------------------------------
    # STEP 3 — OPTICAL FLOW
    # -------------------------------------------------------------------------
    # run_optical_flow retains the RAM-fix design from V4.6:
    #   * bounded number of in-flight frame pairs,
    #   * each completed result is written immediately,
    #   * no all-frame results dictionary,
    #   * velocity video is rendered later from disk-backed PKL files.
    flow_start = time.perf_counter()

    run_optical_flow(
        images,
        output_folder,
        output_pkl,
        output_wxf,
        global_min,
        global_max,
    )

    flow_elapsed = time.perf_counter() - flow_start
    print(f"[ TIMING ] Optical flow + outputs took {flow_elapsed:.2f} s")

    overall_elapsed = time.perf_counter() - overall_start
    print(f"\n[ DONE ] Total pipeline time: {overall_elapsed:.2f} s")
    print(f"[ DONE ] Output folder: {output_folder}")
