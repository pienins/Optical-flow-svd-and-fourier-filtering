"""Input image loading, optional temporal SVD, and normalization."""
from __future__ import annotations
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from tifffile import imread
from tqdm import tqdm


def numeric_sort_key(filename: str) -> int:
    """Extract first integer from filename for natural sorting."""
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else -1


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

