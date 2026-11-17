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
| Read and evaluate uniform test | `benchmarks/evaluators/evaluate_2wcf_uniform_test.py` |
| Read and evaluate Rankine vortex | `benchmarks/evaluators/evaluate_2wcf_rankine_test.py` |

## Code provenance

The modular implementation was uploaded as `2WCF_explicit_refactored(1).zip`. This distribution retains the refactor's source, sample config, and tests, rather than silently merging missing historical `frame2_polynomial` options from the archived Rankine run. See [Current state](current_state.md) for the version boundary.
