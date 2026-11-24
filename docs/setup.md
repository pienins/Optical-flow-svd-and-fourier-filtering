# Setup and running the current code

## Getting the repository

Clone the repository and fetch the large files with Git LFS:

```bash
git clone https://github.com/pienins/Optical-flow-svd-and-fourier-filtering.git
cd Optical-flow-svd-and-fourier-filtering
git lfs install
git lfs pull
```

Do not use GitHub's "Download ZIP". A ZIP download contains small Git LFS placeholder text files instead of the actual `.7z` archives.

## Requirements

Python 3.10+ is a reasonable starting target, but no cross-platform supported-version matrix was supplied. Use the package versions tested in your own environment when publishing. The submitted `requirements.txt` lists: NumPy, OpenCV (`opencv-python`), SciPy, tifffile, pandas, Matplotlib, imageio and tqdm. `wolframclient` is optional only when `outputs.WriteWXF=false`; `pytest` is required for tests.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The 11 tests in `tests/` passed when last run (24 Nov 2026, Linux, Python 3). They are a smoke/regression check, including one small generate → pipeline → evaluate benchmark run, not a numerical verification across all settings.

The Windows scripts `flowenv_setup.bat` and `setup_fieldenv.bat` create Conda environments. **`setup_fieldenv.bat` deletes any existing Conda environment named `fieldenv`**, so read it before running it.

## Input sequence

A single directory containing sorted 2-D grayscale `.tif` or `.tiff` frames of equal shape, at least two frames. The loader uses the first integer in each filename for sorting. Avoid ambiguous numeric names and nonuniform time intervals. The program computes **displacement in pixels per frame pair**, not velocity in SI units unless `pixel_size / frame_period` is applied externally.

## Configuration

Copy the supplied `config.json`, edit only the JSON copy, and use absolute paths or paths relative to the JSON file. **The defaults in the submitted config refer to an external MMC experiment**, so do not expect it to work unmodified on a new computer. The key groups are `paths`, `run`, `preprocessing`, `farneback`, `information_solver`, `consistency`, `confidence`, `visualization`, `diagnostics`, and `outputs`. The parser rejects unknown/missing keys and invalid basic values.

Run:

```bash
python run.py --config /path/to/config.json
```

Set `outputs.WriteWXF=false` if Wolfram is not required; set `outputs.SaveVelocityVideo=false` if no video is needed. Set `diagnostics.SaveValidationData=true` to obtain the sampled NPZ arrays used by the evaluation scripts. `run.MaxFramePairs=5` means five consecutive pairs (0→1 ... 4→5); `null` processes all pairs. `UseParallelFlow=true` allows concurrent pair processing with bounded in-flight tasks; the **input stack is still kept in RAM**.

## Important run safety

`run.runID` and sequence name determine the output directory. Existing nonempty directories raise an error unless `outputs.OverwriteExistingRun=true`; overwriting **clears that run directory**. Always use a new run ID for archival research. The new refactor creates `run_manifest.json` and copies `parameters.json` to the run directory. Include a Git commit hash separately when publishing, as the current manifest does not record one.

A run directory contains `pkl/`, `validation/` (when `diagnostics.SaveValidationData=true`), `flow_qc_metrics.csv`, `parameters.json` and `run_manifest.json`; see [Output formats](output_formats.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| No input images | `paths.input_images` refers to directory containing TIFFs, not parent run folder |
| Unknown JSON key | All required groups and parameters must be present; derive from sample config |
| Existing run folder | Increase `run.runID`, or explicitly overwrite disposable results |
| WXF import failure | Disable `outputs.WriteWXF`, or install `wolframclient` |
| Missing plots/video | Check `outputs` and `diagnostics` switches; they may be intentionally disabled |
| Low RAM | Disable SVD, reduce frame size or count, limit workers; the input stack itself is loaded entirely |
| Reproducing archived Rankine | Its legacy `frame2_polynomial` mode is unavailable in the modular source supplied |
