# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 00:13:42 2025

@main_dev: Ansis_Z
@co_dev: Mihails_B

"""

import os
import re
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
import wolframclient.serializers as wxf
from tqdm import tqdm  


# ===============================
# === UTILS ===
# ===============================

def numeric_sort_key(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


# =====================
# === SVD FILTERING ===
# =====================

def svd_filter_images(image_folder, remove_modes=[0, 12, 14, 15, 18, 19, 26, 27, 39, 41, 46, 48, 53, 66, 89, 99], do_plots=True):
    print("\n Loading TIFF images...")
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tiff')],
                         key=numeric_sort_key)
    images = [imread(os.path.join(image_folder, f)) for f in tqdm(image_files, desc="Reading TIFFs")]
    volume = np.stack(images, axis=0).astype(np.float32)  # shape: (T, H, W)
    T, H, W = volume.shape
    print("Loaded image stack shape:", volume.shape)

    # Flatten (T, H*W) for SVD
    print("\n Performing SVD...")
    X = volume.reshape(T, -1)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Optionally plot first modes
    if do_plots:
        plt.figure(figsize=(8, 3))
        for i in range(3):
            plt.plot(U[:, i], label=f"Mode {i}")
        plt.title("Temporal SVD Modes")
        plt.xlabel("Frame")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Frequency analysis
        num_modes_to_plot = 10
        fig, axs = plt.subplots(num_modes_to_plot, 1, figsize=(10, 2 * num_modes_to_plot), sharex=True)
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
    print("\n Removing selected modes...")
    recon = np.zeros_like(X)
    for i in tqdm(remove_modes, desc="Removing modes"):
        recon += np.outer(U[:, i], S[i] * Vt[i, :])
    X_filtered = X - recon
    filtered_volume = X_filtered.reshape(T, H, W)
    
    print(X_filtered.min(), X_filtered.max())

    
    return filtered_volume, image_files


# ====================
# === OPTICAL FLOW ===
# ====================
def normalize_frame(frame):
    frame = frame - np.min(frame)              # shift min to 0
    frame = frame / np.max(frame)              # scale to [0,1]
    frame_uint8 = (frame * 255).astype(np.uint8)
    return frame_uint8

def run_optical_flow(images, image_files, output_folder_global, runID=1,
                     ArrowDensityStep=3, ConsistencyErrorThreshold=1,
                     VideoFPS=5):

    print("\n Starting optical flow analysis...")

    # Create output dirs
    output_folder = os.path.join(output_folder_global, os.path.basename("filtered_stack") + f"_run_{runID}")
    os.makedirs(output_folder, exist_ok=True)
    output_pkl = os.path.join(output_folder, "pkl")
    os.makedirs(output_pkl, exist_ok=True)
    output_wxf = os.path.join(output_folder, "wxf")
    os.makedirs(output_wxf, exist_ok=True)

    video_path = os.path.join(output_folder, "velocity_field.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    frame_size = None

    # Optical flow parameters
    PyramidDownsampleFactor = 0.5
    PyramidLevels = 8
    WindowSize = 10
    IterationCount = 5
    PolyFitKernelSize = 7
    GaussKernelStdev = 2
    FilterFlags = 0

    parameters = {name: eval(name) for name in dir() if name[0].isupper()
                  and name not in ('In', 'Out', 'BytesIO', 'np', 'cv2', 'os', 'plt')}
    with open(os.path.join(output_folder, "parameters.json"), 'w') as fp:
        json.dump(parameters, fp)
    param_df = pandas.DataFrame(list(parameters.items()))
    param_df.to_csv(os.path.join(output_folder, "parameters.csv"), index=False, header=False)

    h, w = images[0].shape

    # Loop over frames with tqdm
    for i in tqdm(range(len(images) - 1), desc="Optical flow frames"):
        prev_img = normalize_frame(images[i])
        next_img = normalize_frame(images[i + 1])

        # Forward flow
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img,
                                            None, PyramidDownsampleFactor, PyramidLevels, WindowSize,
                                            IterationCount, PolyFitKernelSize, GaussKernelStdev, FilterFlags)
        u, v = flow[..., 0], flow[..., 1]

        # Backward flow for consistency
        flow_bwd = cv2.calcOpticalFlowFarneback(next_img, prev_img,
                                                None, PyramidDownsampleFactor, PyramidLevels, WindowSize,
                                                IterationCount, PolyFitKernelSize, GaussKernelStdev, FilterFlags)

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        coords_x = (grid_x + u).astype(np.float32)
        coords_y = (grid_y + v).astype(np.float32)
        bwd_u_warped = cv2.remap(flow_bwd[..., 0], coords_x, coords_y,
                                 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bwd_v_warped = cv2.remap(flow_bwd[..., 1], coords_x, coords_y,
                                 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        consistency_error = np.sqrt((u + bwd_u_warped) ** 2 + (v + bwd_v_warped) ** 2)
        reliability_mask = consistency_error < ConsistencyErrorThreshold

        # Sampled vectors
        y, x = np.mgrid[ArrowDensityStep//2:h:ArrowDensityStep,
                        ArrowDensityStep//2:w:ArrowDensityStep]
        u_sampled = -1 * u[ArrowDensityStep//2:h:ArrowDensityStep,
                           ArrowDensityStep//2:w:ArrowDensityStep]
        v_sampled = -1 * v[ArrowDensityStep//2:h:ArrowDensityStep,
                           ArrowDensityStep//2:w:ArrowDensityStep]
        reliability_sampled = reliability_mask[ArrowDensityStep//2:h:ArrowDensityStep,
                                               ArrowDensityStep//2:w:ArrowDensityStep]

        valid_indices = reliability_sampled
        x = x[valid_indices]
        y = y[valid_indices]
        u_sampled = u_sampled[valid_indices]
        v_sampled = v_sampled[valid_indices]

        # Save velocity field
        flat_data = {
            'x': x.ravel().astype(float),
            'y': y.ravel().astype(float),
            'u': u_sampled.ravel().astype(float),
            'v': v_sampled.ravel().astype(float)
        }

        pkl_path = os.path.join(output_pkl, f"velocity_field_{i:05d}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(flat_data, f, pickle.HIGHEST_PROTOCOL)
        wxf.export(flat_data, os.path.join(output_wxf, f"velocity_field_{i:05d}.wxf"),
                   target_format='wxf')

        # Plot and save frame
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(prev_img, cmap='gray')
        ax.quiver(x, y, u_sampled, v_sampled, color='lime', scale=50)
        ax.set_title(f'Frame {i} → {i+1}')
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
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

    if video_writer:
        video_writer.release()
    print(f"\n Optical flow MP4 saved to: {video_path}")

#%%
# =====================
# === MAIN PIPELINE ===
# =====================

if __name__ == "__main__":
    # Input/output folders
    image_folder = r"C:\Users\ansis\OneDrive\Desktop\Velocimetry\Velocimetry_pack_11.06.2025\3 Velocimetry\optical_density"
    output_folder_global = r"C:\Users\ansis\OneDrive\Desktop\SMI\test_folder\two_way_farneback_v3_output"

    # Step 1: SVD filtering
    filtered_volume, image_files = svd_filter_images(image_folder, remove_modes=[0], do_plots=False)
    images = [filtered_volume[i] for i in range(filtered_volume.shape[0])]

    # Step 2: Optical flow
    run_optical_flow(images, image_files, output_folder_global, runID=1)

