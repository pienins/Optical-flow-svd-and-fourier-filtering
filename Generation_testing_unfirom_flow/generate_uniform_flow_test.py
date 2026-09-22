from pathlib import Path
import json

import cv2
import numpy as np
from tifffile import imwrite


# =============================================================================
# USER SETTINGS
# =============================================================================
WIDTH = 512
HEIGHT = 512
N_FRAMES = 100

# Approximate number of particles visible in the 512 x 512 camera window.
TARGET_VISIBLE_PARTICLES = 4000

# Exact ground-truth displacement per frame in image coordinates.
# +u = right, +v = down.
TRUE_U = 1   # px/frame
TRUE_V = 0  # px/frame

# Particle image properties.
PARTICLE_SIGMA = 1.1       # Gaussian sigma [px]
PARTICLE_PEAK = 220.0      # isolated-particle peak brightness before uint8 clipping

SEED = 42
RUN_NAME = f"synthetic_uniform_u{TRUE_U}_v{TRUE_V}"

OUTPUT_DIR = Path("E:/Ansis Zivers/Optical Flow/Image_sequences/SYNTHETIC/Uniform_flow")/RUN_NAME
# =============================================================================


FRAME_DIR = OUTPUT_DIR / "frames"
FRAME_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# Render padding lets particles just outside the camera contribute Gaussian tails
# inside the image, avoiding an artificial hard cutoff at the borders.
RENDER_PAD = max(6, int(np.ceil(4.0 * PARTICLE_SIGMA)) + 2)

def format_value_for_filename(value):
    """
    Convert a numerical value into a filename-safe string.

    Examples
    --------
     2.5   -> '2p5'
     1.25  -> '1p25'
     8.0   -> '8'
     0.0   -> '0'
    -2.5   -> 'm2p5'
    """

    value = float(value)

    # Avoid things like -0.0
    if abs(value) < 1e-12:
        value = 0.0

    # Remove unnecessary trailing zeros
    text = f"{abs(value):g}"

    # Decimal point -> p
    text = text.replace(".", "p")

    # Negative -> m prefix
    if value < 0:
        text = "m" + text

    return text

def gaussian_scale_for_peak(sigma: float, requested_peak: float) -> float:
    """Return a fixed multiplier so one centered unit impulse peaks near requested_peak."""
    test = np.zeros((101, 101), dtype=np.float32)
    test[50, 50] = 1.0
    blurred = cv2.GaussianBlur(
        test,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    peak = float(blurred.max())
    if peak <= 0:
        raise RuntimeError("Could not determine Gaussian particle scale.")
    return requested_peak / peak


PARTICLE_SCALE = gaussian_scale_for_peak(PARTICLE_SIGMA, PARTICLE_PEAK)


def render_particles(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Render floating-point particle centres using bilinear splatting followed by
    a fixed Gaussian PSF. No frame-by-frame intensity normalisation is used.
    """
    pad = RENDER_PAD
    canvas_h = HEIGHT + 2 * pad
    canvas_w = WIDTH + 2 * pad

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    xp = x + pad
    yp = y + pad

    x0 = np.floor(xp).astype(np.int32)
    y0 = np.floor(yp).astype(np.int32)

    dx = xp - x0
    dy = yp - y0

    splats = (
        (x0,     y0,     (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0,      dx * (1.0 - dy)),
        (x0,     y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1,  dx * dy),
    )

    for xx, yy, weight in splats:
        inside = (
            (xx >= 0) & (xx < canvas_w) &
            (yy >= 0) & (yy < canvas_h)
        )
        np.add.at(canvas, (yy[inside], xx[inside]), weight[inside])

    canvas = cv2.GaussianBlur(
        canvas,
        (0, 0),
        sigmaX=PARTICLE_SIGMA,
        sigmaY=PARTICLE_SIGMA,
        borderType=cv2.BORDER_CONSTANT,
    )

    canvas *= PARTICLE_SCALE
    canvas = np.clip(canvas, 0.0, 255.0)

    image = canvas[pad:pad + HEIGHT, pad:pad + WIDTH]
    return image.astype(np.uint8)


# =============================================================================
# CREATE AN EXTENDED PARTICLE DOMAIN
# =============================================================================
# The camera sees only WIDTH x HEIGHT, but particles are seeded outside it too.
# This way particles can naturally enter and leave the field of view as the
# uniform flow translates the entire particle field.

total_dx = abs(TRUE_U) * (N_FRAMES - 1)
total_dy = abs(TRUE_V) * (N_FRAMES - 1)

margin_x = int(np.ceil(total_dx)) + RENDER_PAD + 10
margin_y = int(np.ceil(total_dy)) + RENDER_PAD + 10

visible_area = WIDTH * HEIGHT
extended_area = (WIDTH + 2 * margin_x) * (HEIGHT + 2 * margin_y)
particle_density = TARGET_VISIBLE_PARTICLES / visible_area
n_particles_extended = int(np.ceil(particle_density * extended_area))

x_initial = rng.uniform(
    -margin_x,
    WIDTH + margin_x,
    n_particles_extended,
).astype(np.float64)

y_initial = rng.uniform(
    -margin_y,
    HEIGHT + margin_y,
    n_particles_extended,
).astype(np.float64)


# =============================================================================
# GENERATE TIFF SEQUENCE
# =============================================================================
print("Generating synthetic uniform-flow sequence...")
print(f"Image size: {WIDTH} x {HEIGHT}")
print(f"Frames: {N_FRAMES}")
print(f"Ground truth: u={TRUE_U:.6g}, v={TRUE_V:.6g} px/frame")
print(f"GT speed: {np.hypot(TRUE_U, TRUE_V):.6g} px/frame")
print(f"Extended particles: {n_particles_extended}")

for frame_index in range(N_FRAMES):
    x = x_initial + TRUE_U * frame_index
    y = y_initial + TRUE_V * frame_index

    image = render_particles(x, y)

    # .tiff is accepted by the updated 2WCF loader.
    output_path = FRAME_DIR / f"frame_{frame_index:05d}.tiff"
    imwrite(output_path, image, compression=None)


# Full-resolution truth maps are convenient later when we move to nonuniform flows.
u_gt = np.full((HEIGHT, WIDTH), TRUE_U, dtype=np.float32)
v_gt = np.full((HEIGHT, WIDTH), TRUE_V, dtype=np.float32)

np.savez_compressed(
    OUTPUT_DIR / "true_vector_values.npz",
    u_gt=u_gt,
    v_gt=v_gt,
    true_u=np.float32(TRUE_U),
    true_v=np.float32(TRUE_V),
    true_speed=np.float32(np.hypot(TRUE_U, TRUE_V)),
)

parameters = {
    "width": WIDTH,
    "height": HEIGHT,
    "n_frames": N_FRAMES,
    "target_visible_particles": TARGET_VISIBLE_PARTICLES,
    "true_u_px_per_frame": TRUE_U,
    "true_v_px_per_frame": TRUE_V,
    "true_speed_px_per_frame": float(np.hypot(TRUE_U, TRUE_V)),
    "particle_sigma_px": PARTICLE_SIGMA,
    "particle_peak": PARTICLE_PEAK,
    "seed": SEED,
    "coordinate_system": {
        "positive_u": "right",
        "positive_v": "down",
    },
}

with open(OUTPUT_DIR / "parameters.json", "w", encoding="utf-8") as f:
    json.dump(parameters, f, indent=2)

print("\nDone.")
print(f"Frames:       {FRAME_DIR.resolve()}")
print(f"Ground truth: {(OUTPUT_DIR / 'ground_truth.npz').resolve()}")
print(f"Parameters:   {(OUTPUT_DIR / 'parameters.json').resolve()}")
