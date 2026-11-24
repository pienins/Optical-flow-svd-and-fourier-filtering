# Historical research material in this repository

*Historical record; reviewed 24 November 2026.* These items predate the modular `run.py` / `two_wcf/` implementation and are kept unchanged so that earlier work can be traced. Keeping a legacy script does not imply that its outputs match the current algorithm; benchmark outputs must be linked to the source version that produced them.

| Item | What it is | Notes |
|---|---|---|
| `legacy/2WCF-SVD_v4_4.py` | Historical monolithic OpenCV-based Farnebäck + 2WCF with optional SVD | Moved from the repository root with `git mv`, so its Git history is kept |
| `Generation_testing_unfirom_flow/` | Original uniform-flow generator, evaluator and validation PDF | Folder name kept as is (including the spelling) so historical links still work; the maintained, config-driven versions are in `benchmarks/homogeneous_constant_flow/` |
| `Artifact_mode_detection/` | Mathematica notebook, Python Fourier/SVD mode analysis and WXF results | Where temporal SVD modes are analysed and `KillIDs` are chosen (`methods.md` §1.1) |
| `OFV_field_postprocessing_v2.nb` | Mathematica field post-processing | Kept in place |
| `experimental_data/sequences/` | Four `.7z` experimental image sequences | Git LFS; moved from `Test_image_sequences/` on 24 Nov 2026 |
| `experimental_data/outputs/` | Six `.7z` outputs computed from three of those sequences | Git LFS; moved from `Example_2WCF-SVD_outputs/` on 24 Nov 2026 |
| `flowenv_setup.bat`, `setup_fieldenv.bat` and matching `.txt` | Windows Conda environment helpers | **`setup_fieldenv.bat` deletes any existing Conda environment named `fieldenv`** before creating it |
| `.gitattributes` | Git LFS rule | Tracks `*.7z` only |

A GitHub "Download ZIP" cannot recover the LFS archives: it contains ~130-byte pointer files instead. Clone with Git LFS and run `git lfs pull` ([setup](../setup.md)).
