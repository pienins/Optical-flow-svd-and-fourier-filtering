"""Generate a synthetic homogeneous (uniform) translation benchmark.

Every tracer particle moves by the same exact displacement (u, v) per frame,
so the ground truth is spatially constant. Coordinates: +u right, +v down;
units px/frame.

Usage (from any directory):
    python benchmarks/homogeneous_constant_flow/generate.py [--config PATH]

All settings and paths are read from config.json next to this script.
Relative paths in the config are resolved against the config file.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tifffile import imwrite

DEFAULT_CONFIG = Path(__file__).with_name("config.json")

CONFIG_KEYS = {
    "paths": {"generated_root", "pipeline_output_dir"},
    "generate": {
        "width", "height", "n_frames", "target_visible_particles",
        "true_u_px_per_frame", "true_v_px_per_frame",
        "particle_sigma_px", "particle_peak", "seed", "overwrite_existing",
    },
    "evaluate": {"interior_border_px"},
}


# =============================================================================
# CONFIGURATION
# =============================================================================
def load_config(path=DEFAULT_CONFIG):
    """Read and validate the benchmark JSON; return it with paths resolved."""
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    if set(cfg) != set(CONFIG_KEYS):
        raise ValueError(f"{path}: top-level groups must be {sorted(CONFIG_KEYS)}")
    for group, keys in CONFIG_KEYS.items():
        if set(cfg[group]) != keys:
            missing = sorted(keys - set(cfg[group]))
            unknown = sorted(set(cfg[group]) - keys)
            raise ValueError(f"{path}: group '{group}' missing {missing}, unknown {unknown}")

    g = cfg["generate"]
    for key in ("width", "height", "n_frames", "target_visible_particles"):
        if not isinstance(g[key], int) or isinstance(g[key], bool) or g[key] < 1:
            raise ValueError(f"generate.{key} must be a positive integer.")
    if g["n_frames"] < 2:
        raise ValueError("generate.n_frames must be at least 2.")
    if not isinstance(g["seed"], int) or isinstance(g["seed"], bool):
        raise ValueError("generate.seed must be an integer.")
    for key in ("true_u_px_per_frame", "true_v_px_per_frame",
                "particle_sigma_px", "particle_peak"):
        if not isinstance(g[key], (int, float)) or isinstance(g[key], bool):
            raise ValueError(f"generate.{key} must be a number.")
    if g["particle_sigma_px"] <= 0 or g["particle_peak"] <= 0:
        raise ValueError("generate.particle_sigma_px and particle_peak must be positive.")
    if not isinstance(g["overwrite_existing"], bool):
        raise ValueError("generate.overwrite_existing must be true or false.")

    border = cfg["evaluate"]["interior_border_px"]
    if not isinstance(border, int) or isinstance(border, bool) or border < 0:
        raise ValueError("evaluate.interior_border_px must be a non-negative integer.")

    for key, value in cfg["paths"].items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"paths.{key} must be a non-empty string.")
        candidate = Path(value).expanduser()
        cfg["paths"][key] = candidate if candidate.is_absolute() else (path.parent / candidate).resolve()

    cfg["_config_path"] = path
    return cfg


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


def sequence_name(generate_cfg):
    """Folder name of one generated sequence, e.g. synthetic_uniform_u8_vm20."""
    u = format_value_for_filename(generate_cfg["true_u_px_per_frame"])
    v = format_value_for_filename(generate_cfg["true_v_px_per_frame"])
    return f"synthetic_uniform_u{u}_v{v}"


def sequence_dir(cfg):
    """Directory holding frames/, true_vector_values.npz and parameters.json."""
    return cfg["paths"]["generated_root"] / sequence_name(cfg["generate"])


# =============================================================================
# RENDERING
# =============================================================================
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


def render_particles(x, y, width, height, sigma, scale, pad):
    """
    Render floating-point particle centres using bilinear splatting followed by
    a fixed Gaussian PSF. No frame-by-frame intensity normalisation is used.
    """
    canvas_h = height + 2 * pad
    canvas_w = width + 2 * pad

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
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )

    canvas *= scale
    canvas = np.clip(canvas, 0.0, 255.0)

    image = canvas[pad:pad + height, pad:pad + width]
    return image.astype(np.uint8)


# =============================================================================
# GENERATION
# =============================================================================
def prepare_output(output_dir, frame_dir, overwrite):
    """Refuse to mix new frames with an older sequence unless overwrite is set."""
    existing = sorted(frame_dir.glob("frame_*.tif*")) if frame_dir.exists() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"{frame_dir} already contains {len(existing)} frames. Set "
            "generate.overwrite_existing=true to replace them, or change the "
            "flow parameters / paths.generated_root."
        )
    for old in existing:
        old.unlink()
    frame_dir.mkdir(parents=True, exist_ok=True)


def generate(cfg):
    g = cfg["generate"]
    width, height, n_frames = g["width"], g["height"], g["n_frames"]
    true_u, true_v = g["true_u_px_per_frame"], g["true_v_px_per_frame"]
    sigma, peak = g["particle_sigma_px"], g["particle_peak"]

    output_dir = sequence_dir(cfg)
    frame_dir = output_dir / "frames"
    prepare_output(output_dir, frame_dir, g["overwrite_existing"])

    rng = np.random.default_rng(g["seed"])

    # Render padding lets particles just outside the camera contribute Gaussian tails
    # inside the image, avoiding an artificial hard cutoff at the borders.
    render_pad = max(6, int(np.ceil(4.0 * sigma)) + 2)
    particle_scale = gaussian_scale_for_peak(sigma, peak)

    # -------------------------------------------------------------------------
    # Extended particle domain: the camera sees only width x height, but
    # particles are seeded outside it too, so they can enter and leave the
    # field of view as the uniform flow translates the entire particle field.
    # -------------------------------------------------------------------------
    total_dx = abs(true_u) * (n_frames - 1)
    total_dy = abs(true_v) * (n_frames - 1)

    margin_x = int(np.ceil(total_dx)) + render_pad + 10
    margin_y = int(np.ceil(total_dy)) + render_pad + 10

    visible_area = width * height
    extended_area = (width + 2 * margin_x) * (height + 2 * margin_y)
    particle_density = g["target_visible_particles"] / visible_area
    n_particles_extended = int(np.ceil(particle_density * extended_area))

    x_initial = rng.uniform(-margin_x, width + margin_x, n_particles_extended).astype(np.float64)
    y_initial = rng.uniform(-margin_y, height + margin_y, n_particles_extended).astype(np.float64)

    print("Generating synthetic uniform-flow sequence...")
    print(f"Image size: {width} x {height}")
    print(f"Frames: {n_frames}")
    print(f"Ground truth: u={true_u:.6g}, v={true_v:.6g} px/frame")
    print(f"GT speed: {np.hypot(true_u, true_v):.6g} px/frame")
    print(f"Extended particles: {n_particles_extended}")

    for frame_index in range(n_frames):
        x = x_initial + true_u * frame_index
        y = y_initial + true_v * frame_index
        image = render_particles(x, y, width, height, sigma, particle_scale, render_pad)
        imwrite(frame_dir / f"frame_{frame_index:05d}.tiff", image, compression=None)

    # Full-resolution truth maps, so evaluation can look up truth at any sample.
    u_gt = np.full((height, width), true_u, dtype=np.float32)
    v_gt = np.full((height, width), true_v, dtype=np.float32)

    gt_path = output_dir / "true_vector_values.npz"
    np.savez_compressed(
        gt_path,
        u_gt=u_gt,
        v_gt=v_gt,
        true_u=np.float32(true_u),
        true_v=np.float32(true_v),
        true_speed=np.float32(np.hypot(true_u, true_v)),
    )

    parameters = {
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "target_visible_particles": g["target_visible_particles"],
        "true_u_px_per_frame": true_u,
        "true_v_px_per_frame": true_v,
        "true_speed_px_per_frame": float(np.hypot(true_u, true_v)),
        "particle_sigma_px": sigma,
        "particle_peak": peak,
        "seed": g["seed"],
        "coordinate_system": {
            "positive_u": "right",
            "positive_v": "down",
        },
    }
    params_path = output_dir / "parameters.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=2)

    print("\nDone.")
    print(f"Frames:       {frame_dir.resolve()}")
    print(f"Ground truth: {gt_path.resolve()}")
    print(f"Parameters:   {params_path.resolve()}")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="benchmark config.json")
    args = parser.parse_args(argv)
    generate(load_config(args.config))


if __name__ == "__main__":
    main()
