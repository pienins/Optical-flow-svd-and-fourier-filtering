# Input, output and result formats

| Artifact | Producer | Contents / interpretation |
|---|---|---|
| `.tif` / `.tiff` | Image generator or camera | Each grayscale input image; natural-numeric sorting |
| `parameters.json` (grouped) | Modular pipeline | Exact config snapshot with all groups and paths |
| `run_manifest.json` | Modular pipeline | Input filenames, processed indices, normalization and elapsed time; **currently no source commit hash** |
| `pkl/velocity_field_00000.pkl` | Pipeline | Per-pair retained arrays `x`, `y`, `u`, `v` (pickle; only load trusted files) |
| `wxf/velocity_field_00000.wxf` | Optional pipeline | Wolfram-compatible saved retained vectors |
| `validation/validation_00000.npz` | Pipeline when enabled | Sampled raw/processed/output displacement, masks, error and confidence |
| `flow_qc_metrics.csv` | Pipeline | Frame-pair median consistency residual, scaled MAD, thresholds, counts and retention |
| `diagnostic_maps/` | Optional pipeline | Local MAD, sigma and confidence arrays and videos |
| `explicit_flow_diagnostics/` | Optional pipeline | Raw dense pyramid-flow maps, information matrix diagnostics and streamlines |
| `velocity_field.mp4` | Optional pipeline | Video from retained flow fields (uniform speed scale across frames) |
| `true_vector_values.npz` | Synthetic generator | External truth maps; **not bundled in the historical `.7z` uploads** |

## Validation NPZ schema

`x`, `y` are sampled Cartesian image coordinates (2-D grids); `u_raw`, `v_raw` are sampled pre-filter Farnebäck estimates; `u_processed`, `v_processed` are sampled post-processing values (identical to raw in hard mode); `u_output`, `v_output` are the values after the optional `VelocityInversion`; `valid_mask` flags retained vectors; `consistency_mask` records hybrid-pass decisions; `consistency_error` records forward/backward residual magnitude; `confidence` is the confidence diagnostic. The `u_output` and `v_output` grids may include **rejected candidates**; apply `valid_mask` when computing retained-output statistics. The full vectors in `pkl/` contain only retained positions.

All displacement components are in **pixels per frame pair**. A 2-D image does not independently determine the third component of a 3-D velocity field.

### Note on the uploaded archives

Archived `parameters.json` files are **flat**, unlike the grouped config written by the new refactor. They do not contain the modular `run_manifest.json` and must be interpreted with their original code version. The Rankine archive additionally records a warp mode unsupported in this refactor.
