# Code and mathematical-operation map

| Operation | Module and function |
|---|---|
| Read TIFF stack and optional SVD | `two_wcf/image_io.py`: `load_images_without_svd`, `svd_filter_images` |
| Normalize image intensities | `two_wcf/image_io.py`: `normalize_frame` |
| Build actual Gaussian pyramid | `two_wcf/solver.py`: `build_farneback_visualization_pyramid` |
| Gaussian-weighted 2-D polynomial fit | `two_wcf/solver.py`: `make_quadratic_fit_kernels`, `quadratic_polynomial_expansion` |
| Warp frame-2 intensities | `two_wcf/solver.py`: `warp_second_frame` |
| Sample frame-2 coefficients with optional edge taper | `two_wcf/solver.py`: `sample_farneback_coefficients` |
| Integrate normal equations, ridge and solve | `two_wcf/solver.py`: `farneback_residual_increment` |
| Coarse-to-fine iterative solver | `two_wcf/solver.py`: `calc_optical_flow_farneback_explicit` |
| Forward/backward field, consistency, sample/reject | `two_wcf/consistency.py`: `compute_flow_pair` |
| Local sigma and confidence smoothing | `two_wcf/consistency.py`: `calculate_local_sigma`, `confidence_weighted_smoothing` |
| Load a full run configuration | `two_wcf/config.py`: `load_config`, `validate` |
| Run orchestration, disk-backed field saving | `two_wcf/pipeline.py`: `run`, `_process_pairs` |
| Write PKL/WXF/CSV/NPZ figures | `two_wcf/outputs.py` |
| Make velocity/diagnostic plots and videos | `two_wcf/plots.py`, `two_wcf/videos.py`, `two_wcf/diagnostics.py` |
| Benchmark metrics (EPE, angular error, safe statistics) | `benchmarks/common/metrics.py` |
| Generate uniform-flow benchmark and ground truth | `benchmarks/homogeneous_constant_flow/generate.py`: `generate`, `render_particles` |
| Evaluate a run against uniform-flow ground truth | `benchmarks/homogeneous_constant_flow/evaluate.py`: `evaluate` |
| Read and evaluate Rankine vortex (not yet migrated) | `benchmarks/evaluators/evaluate_2wcf_rankine_test.py` |

## Execution order, memory and output timing

`pipeline.run` executes one sequence in this order:

1. Load all TIFF frames (`image_io.load_images_without_svd`), or load them and apply temporal SVD (`image_io.svd_filter_images`). The whole input stack stays in memory for the run, and with SVD enabled so does the $T\times HW$ matrix.
2. Take the global intensity minimum and maximum over all (filtered) frames. Every later step normalizes with this single range.
3. Optionally write the pyramid resolution table and pyramid image diagnostics.
4. Process frame pairs `(0,1), (1,2), …`, limited by `run.MaxFramePairs` (`pipeline._process_pairs`). Each pair is one call to `consistency.compute_flow_pair`, which solves forward and backward flow and applies the consistency selection.
   - `UseParallelFlow = true` runs pairs in a thread pool. At most `MaxFlowWorkers` pairs are in flight at once (default: CPU count − 1), so memory for dense flow fields stays bounded. Pairs may finish out of order; files are named by pair index and the QC records are sorted before writing.
   - As soon as a pair finishes, its retained vectors (`outputs.save_frame_field`), optional validation NPZ (`outputs.save_validation`) and optional diagnostic maps are written to disk, and the dense arrays are released. Only scalar QC records and the running minimum/maximum of retained speed are kept in memory.
5. Render the velocity video in a second pass that reads the saved fields back from disk (`videos.render_velocity_video`). All frames share one color scale, the minimum and maximum retained speed over the whole run, so colors are comparable between frames.
6. Render optional diagnostic-map videos, then write `flow_qc_metrics.csv` and the optional QC plot.
7. For each pair in `DiagnosticFramePairs`, run the forward solver again, sequentially, to capture per-level flow and information-matrix diagnostics (`diagnostics.save_explicit_flow_diagnostics`). This repeats work already done in step 4 but keeps the main loop free of per-level bookkeeping.
8. Write `run_manifest.json`.

See [Status and limitations](status_and_limitations.md) for what is and is not implemented.
