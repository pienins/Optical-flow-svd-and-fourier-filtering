# Existing historical research material in `main`

This page inventories material in the user-supplied `Optical-flow-svd-and-fourier-filtering-main.zip`. No historical script or data archive was modified to create the integration overlay.

| Existing item | Current meaning | Merge treatment |
|---|---|---|
| `2WCF-SVD_v4_4.py` | Historical monolithic OpenCV-based Farnebäck + 2WCF and optional SVD | `git mv` to `legacy/2WCF-SVD_v4_4.py` in a dedicated integration commit |
| `Generation_testing_unfirom_flow/` | Original uniform test generator, validator, validation PDF | Preserve existing spelling/path to avoid breaking historical links; the new `benchmarks/` holds separate copies and a new portable mini example |
| `Artifact_mode_detection/` | Mathematica notebook, Fourier/SVD Python analysis and WXF result data | Preserve original path, explain provenance later |
| `OFV_field_postprocessing_v2.nb` | Mathematica postprocessing | Preserve in place |
| `Test_image_sequences/` | Four `.7z` example image sets | Preserve original LFS files and their Git history |
| `Example_2WCF-SVD_outputs/` | Six `.7z` archived outputs | Preserve original LFS files and their Git history |
| `flowenv_setup.bat`, `setup_fieldenv.bat`, corresponding `.txt` | Windows-era environment helpers | Preserve unchanged; **`setup_fieldenv.bat` deletes any pre-existing Conda environment named `fieldenv`** before creating it |
| `.gitattributes` | Git LFS tracking of `*.7z` | Preserve as-is; do not replace with source-ZIP LFS pointer placeholders |

The current modular `run.py`/`two_wcf/` is intended for new runs. Preservation of a legacy script does not imply its outputs match the current algorithm. Benchmark outputs must link to the correct historical source commit or original script.

The source download ZIP cannot be used to recover LFS archive contents: most `.7z` entries in it are ~130-byte Git LFS pointers. Clone with Git LFS and run `git lfs pull` to retrieve them.
