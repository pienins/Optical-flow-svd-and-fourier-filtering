# Experimental data

Real experimental image sequences and optical-flow outputs computed from them. **There is no ground truth for these data**, so they show how the method behaves on real images but cannot measure its accuracy. Accuracy is measured with the synthetic cases in [`benchmarks/`](../benchmarks/).

All files are `.7z` archives stored with Git LFS. After cloning, run `git lfs pull` to download them ([setup](../docs/setup.md)).

## `sequences/` — input image sequences

| Archive | Size |
|---|---:|
| `RBC_corrected_images_3D_filtered_TV_Gaussian_consRel_0.0175.7z` | 65 MB |
| `RBC_corrected_images_3D_filtered_TV_Gaussian_consRel_0.02.7z` | 64 MB |
| `optical_density_3D_filtered_TV_Gaussian_consRel_0.05.7z` | 3.6 MB |
| `optical_density_3D_filtered_TV_Laplacian_consRel_1.15.7z` | 3.2 MB |

## `outputs/` — results computed from those sequences

| Archive | Input sequence | Size |
|---|---|---:|
| `RBC_…_consRel_0.0175_run_12_v4-4.7z` | `RBC_…_consRel_0.0175` | 25 MB |
| `RBC_…_consRel_0.02_run_1.7z` | `RBC_…_consRel_0.02` | 47 MB |
| `RBC_…_consRel_0.02_run_2.7z` | `RBC_…_consRel_0.02` | 47 MB |
| `optical_density_…_Laplacian_consRel_1.15_run_1.7z` | `optical_density_…_Laplacian_consRel_1.15` | 16 MB |
| `optical_density_…_Laplacian_consRel_1.15_run_2.7z` | `optical_density_…_Laplacian_consRel_1.15` | 16 MB |
| `optical_density_…_Laplacian_consRel_1.15_run_3.7z` | `optical_density_…_Laplacian_consRel_1.15` | 16 MB |

The input sequence is inferred from the file names. Only `run_12_v4-4` records its solver version (`legacy/2WCF-SVD_v4_4.py`); the version that produced the other runs is not recorded. None of these outputs was produced by the current `two_wcf/` code. `optical_density_…_Gaussian_consRel_0.05` has no archived output.
