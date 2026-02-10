# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:50:08 2025

@author: Ansis Emils Zivers
@co-author: Mihails Birjukovs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import re
from skimage.metrics import mean_squared_error
from scipy.fft import fft, fftfreq


from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export

#%%

image_folder = r"D:\Science\NMI\MHD Project\Image Processing\Optical Flow\Image_sequences\optical_density_3D_filtered_TV_Laplacian_consRel_1.15"

def numeric_sort_key(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tiff')],
                     key=numeric_sort_key)

images = [imread(os.path.join(image_folder, f)) for f in image_files]
volume = np.stack(images, axis=0).astype(np.float32)  # shape: (T, H, W)
T, H, W = volume.shape
print("Loaded image stack shape:", volume.shape)

# SVD
X = volume.reshape(T, -1)  # shape: (T, H*W)
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# first few modes
plt.figure(figsize=(8, 3))
for i in range(3):
    plt.plot(U[:, i], label=f"Mode {i}")
plt.title("Temporal SVD Modes")
plt.xlabel("Frame")
plt.legend()
plt.tight_layout()
plt.show()


# frequency analysis
num_modes_to_plot = 100
fft_results = []

fig, axs = plt.subplots(num_modes_to_plot, 1, figsize=(10, 2 * num_modes_to_plot), sharex=True)
fig.suptitle("FFT of Temporal SVD Modes")

for i in range(num_modes_to_plot):
    signal = U[:, i]
    N = len(signal)
    yf = np.abs(fft(signal - np.mean(signal)))
    xf = fftfreq(N, d=1)[:N // 2]  # Assume 1 frame per unit time
    axs[i].plot(xf, yf[:N // 2])
    axs[i].set_ylabel(f"Mode {i}")
    axs[i].grid(True)
    fft_results.append((xf, yf[:N // 2]))

axs[-1].set_xlabel("Frequency (1/frame)")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
#%%

remove_modes = [0, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99] #+ list(range(15, 100))

recon = np.zeros_like(X)
for i in remove_modes:
    recon += np.outer(U[:, i], S[i] * Vt[i, :])
X_filtered = X - recon
filtered_volume = X_filtered.reshape(T, H, W)


avg_intensity_before = volume.mean(axis=(1, 2))
avg_intensity_after = filtered_volume.mean(axis=(1, 2))

plt.figure(figsize=(8, 4))
plt.plot(avg_intensity_before, label="Before Filtering")
plt.plot(avg_intensity_after, label="After Filtering")
plt.title("Average Intensity Over Time")
plt.xlabel("Frame")
plt.ylabel("Mean Intensity")
plt.legend()
plt.tight_layout()
plt.show()

mse_before = [mean_squared_error(volume[0], volume[i]) for i in range(T)]
mse_after = [mean_squared_error(filtered_volume[0], filtered_volume[i]) for i in range(T)]

plt.figure(figsize=(8, 4))
plt.plot(mse_before, label="Before Filtering")
plt.plot(mse_after, label="After Filtering")
plt.title("Frame-wise MSE vs First Frame")
plt.xlabel("Frame")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

#%%


# FFT of Average Intensity Over Time
signal = avg_intensity_before - np.mean(avg_intensity_before)  # remove DC
N = len(signal)
yf = np.abs(fft(signal))
xf = fftfreq(N, d=1)[:N // 2]  # Assuming 1 frame per time unit

plt.figure(figsize=(8, 4))
plt.plot(xf, yf[:N // 2])
plt.title("FFT of Average Intensity Over Time (Before Filtering)")
plt.xlabel("Frequency (1/frame)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
#%% Cancer(ish) further...

# Set and create the output directory
output_folder = r"D:\Science\NMI\MHD Project\Image Processing\Optical Flow\Auto-detect Artifact Modes\Fourier_SVD_filtering_tests"
os.makedirs(output_folder, exist_ok=True)

#   WXF - TEMPORAL MODES

# Prepare data: list of {xf, yf} pairs for each mode
fft_export_data = []
for i, (xf_i, yf_i) in enumerate(fft_results):
    # Store each as a 2-column list for easier plotting in Mathematica
    fft_export_data.append(np.column_stack((xf_i, yf_i)))

# Export to WXF
wxf_path = os.path.join(output_folder, "temporal_mode_ffts.wxf")
with open(wxf_path, "wb") as f:
    export(fft_export_data, f, target_format="wxf")

print(f"Exported FFTs of temporal modes to: {wxf_path}")
#%%   WXF - FFT AVG INT OVER TIME

# Wrap in List[] to ensure correct serialization
fft_data = [ [float(xf[i]), float(yf[i])] for i in range(len(xf)) ]

with WolframLanguageSession() as session:
    session.evaluate(wl.Export(
        os.path.join(output_folder, "fft_avg_intensity.wxf"),
        wl.List(*[ wl.List(*pair) for pair in fft_data ]),
        "WXF"
    ))
#%%   WXF - MSE VS FIRST FRAME
mse_before_arr = np.array(mse_before)
mse_after_arr = np.array(mse_after)

# Frame indices (0-based)
frames = np.arange(len(mse_before_arr))

# Stack into one array: columns = [frame, mse_before, mse_after]
mse_data = np.stack([frames, mse_before_arr, mse_after_arr], axis=1)

mse_wxf_path = os.path.join(output_folder, "mse_timeseries.wxf")
export(mse_data, mse_wxf_path)

print(f"Saved MSE time series data to: {mse_wxf_path}")


#%%  OPTIONAL -- filtered image export for inspection


def normalize_image_numpy(img):
    ar = np.array(img).astype(np.float32)
    mn = np.min(ar)
    mx = np.max(ar)
    return (ar - mn) * (1.0 / (mx - mn))


# saving
output_image_folder = os.path.join(output_folder, "filtered_images")
os.makedirs(output_image_folder, exist_ok=True)

for i in range(T):
    out_path = os.path.join(output_image_folder, f"filtered_frame_{i:04d}.tiff")
    imwrite(out_path, normalize_image_numpy(filtered_volume[i]))

print(f"Saved {T} filtered frames to: {output_folder}")